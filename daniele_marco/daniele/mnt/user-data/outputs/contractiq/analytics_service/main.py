"""
contractiq/analytics_service/main.py

Analytics Service — powers all 5 dashboard views + proactive alerts.

Dashboards:
  1. Clienti Strategici    — revenue concentration (Pareto)
  2. Contratti Critici     — expiry timeline + priority matrix
  3. Condizioni Commerciali — discount/non-standard terms map
  4. Performance Prodotti  — ARR growth by product/service
  5. Clausole a Rischio    — risk heatmap with clause details

Proactive Alerts (Disruptive Move #3):
  - Called by DSPy alert agent
  - Also pre-computed on schedule via Redis
"""

import json
import logging
import os
from datetime import datetime, date, timedelta
from contextlib import asynccontextmanager
from typing import Optional


def _as_list(val, default=None) -> list:
    """Safely coerce a value to list. Handles both native lists and JSON strings."""
    if default is None:
        default = []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else default
        except (json.JSONDecodeError, TypeError):
            return default
    return default

import redis.asyncio as redis
import sqlalchemy as sa
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Application lifespan: connects Redis."""
    global redis_client
    redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True
    )
    logger.info("Analytics service ready")
    yield

app = FastAPI(title="ContractIQ Analytics", version="1.0.0", lifespan=lifespan)

engine = create_async_engine(
    os.getenv("POSTGRES_URL", "").replace("postgresql://", "postgresql+asyncpg://")
)
redis_client = None


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(sa.String(36))
    client_id: Mapped[str] = mapped_column(sa.String(100))
    filename: Mapped[str] = mapped_column(sa.String(255))
    extracted_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    contract_score: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


# Startup gestito dal lifespan context manager (vedi sopra).


@app.get("/health")
async def health():
    return {"status": "ok", "service": "analytics_service"}


@app.get("/portfolio/{client_id}")
async def get_portfolio(client_id: str):
    """
    Returns all 5 dashboard datasets in one call.
    Results cached in Redis for 5 minutes.
    """
    cache_key = f"portfolio:{client_id}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    async with AsyncSession(engine) as db:
        result = await db.execute(
            sa.select(Document)
            .where(Document.client_id == client_id)
            .where(Document.extracted_json.isnot(None))
        )
        docs = result.scalars().all()

    contracts = []
    for doc in docs:
        try:
            analysis = json.loads(doc.extracted_json or "{}")
            extraction = analysis.get("extraction", {})
            scoring = analysis.get("scoring", {})
            contracts.append({
                "id": doc.id,
                "filename": doc.filename,
                "contract_score": doc.contract_score or 0,
                **extraction,
                "scoring": scoring,
                "clausola_gemella": analysis.get("clausola_gemella", []),
            })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Skipping document {doc.id}: {e}")
            continue

    portfolio = {
        "dashboard_1_strategic_clients": _strategic_clients(contracts),
        "dashboard_2_critical_contracts": _critical_contracts(contracts),
        "dashboard_3_commercial_conditions": _commercial_conditions(contracts),
        "dashboard_4_product_performance": _product_performance(contracts),
        "dashboard_5_risk_clauses": _risk_clauses(contracts),
        "summary": _portfolio_summary(contracts),
    }

    # Cache for 5 minutes
    if redis_client:
        await redis_client.setex(cache_key, 300, json.dumps(portfolio))

    return portfolio


def _strategic_clients(contracts: list) -> dict:
    """Dashboard 1: Who generates the most revenue? Concentration risk?"""
    total_value = sum(c.get("total_value_eur", 0) or 0 for c in contracts)

    clients = {}
    for c in contracts:
        name = c.get("client_name", "Unknown")
        val = c.get("total_value_eur", 0) or 0
        arr = c.get("annual_recurring_revenue", 0) or 0
        if name not in clients:
            clients[name] = {"value": 0, "arr": 0, "contracts": 0, "score": 0}
        clients[name]["value"] += val
        clients[name]["arr"] += arr
        clients[name]["contracts"] += 1
        clients[name]["score"] = max(clients[name]["score"], c.get("contract_score", 0))

    sorted_clients = sorted(clients.items(), key=lambda x: x[1]["value"], reverse=True)

    # Herfindahl-Hirschman Index for concentration
    if total_value > 0:
        hhi = sum((v["value"] / total_value * 100) ** 2 for _, v in clients.items())
    else:
        hhi = 0

    concentration_risk = "high" if hhi > 2500 else "medium" if hhi > 1000 else "low"

    return {
        "total_portfolio_value": total_value,
        "total_arr": sum(c.get("annual_recurring_revenue", 0) or 0 for c in contracts),
        "client_count": len(clients),
        "herfindahl_index": round(hhi, 1),
        "concentration_risk": concentration_risk,
        "clients": [
            {
                "name": name,
                "value_eur": data["value"],
                "arr_eur": data["arr"],
                "contracts": data["contracts"],
                "revenue_share_pct": round(data["value"] / total_value * 100, 1) if total_value else 0,
                "contract_score": data["score"],
            }
            for name, data in sorted_clients[:20]
        ],
        "top_5_concentration_pct": round(
            sum(v["value"] for _, v in sorted_clients[:5]) / total_value * 100, 1
        ) if total_value else 0,
    }


def _critical_contracts(contracts: list) -> dict:
    """Dashboard 2: Which contracts expire soon and need urgent attention?"""
    today = date.today()

    expiring = []
    for c in contracts:
        end_date_str = c.get("end_date", "")
        if not end_date_str:
            continue
        try:
            end_date = date.fromisoformat(end_date_str)
            days_to_expiry = (end_date - today).days
            if days_to_expiry <= 180:  # Show next 6 months
                urgency = (
                    "critical" if days_to_expiry <= 30 else
                    "high" if days_to_expiry <= 60 else
                    "medium" if days_to_expiry <= 90 else
                    "low"
                )
                expiring.append({
                    "contract_id": c["id"],
                    "client_name": c.get("client_name", "Unknown"),
                    "filename": c.get("filename", ""),
                    "value_eur": c.get("total_value_eur", 0),
                    "end_date": end_date_str,
                    "days_to_expiry": days_to_expiry,
                    "auto_renewal": c.get("auto_renewal", False),
                    "renewal_notice_days": c.get("renewal_notice_days", 0),
                    "urgency": urgency,
                    "contract_score": c.get("contract_score", 0),
                    "action_deadline": (
                        (end_date - timedelta(days=c.get("renewal_notice_days", 30))).isoformat()
                        if c.get("auto_renewal") else None
                    ),
                })
        except (ValueError, TypeError):
            continue

    expiring.sort(key=lambda x: x["days_to_expiry"])
    revenue_at_risk_30 = sum(c["value_eur"] for c in expiring if c["days_to_expiry"] <= 30)
    revenue_at_risk_90 = sum(c["value_eur"] for c in expiring if c["days_to_expiry"] <= 90)

    return {
        "expiring_contracts": expiring,
        "revenue_at_risk_30_days": revenue_at_risk_30,
        "revenue_at_risk_90_days": revenue_at_risk_90,
        "critical_count": sum(1 for c in expiring if c["urgency"] == "critical"),
        "high_count": sum(1 for c in expiring if c["urgency"] == "high"),
    }


def _commercial_conditions(contracts: list) -> dict:
    """Dashboard 3: Where are we applying non-standard discounts or terms?"""
    non_standard = []
    discount_distribution = []
    payment_distribution = []

    for c in contracts:
        discount = c.get("discount_percentage", 0) or 0
        payment_days = c.get("payment_terms_days", 30) or 30

        discount_distribution.append(discount)
        payment_distribution.append(payment_days)

        non_standard_terms = _as_list(c.get("non_standard_terms", []))
        if discount > 20 or payment_days > 60 or non_standard_terms:
            non_standard.append({
                "contract_id": c["id"],
                "client_name": c.get("client_name", "Unknown"),
                "discount_pct": discount,
                "payment_days": payment_days,
                "non_standard_terms": non_standard_terms,
                "alert": _commercial_alert(discount, payment_days),
            })

    avg_discount = sum(discount_distribution) / len(discount_distribution) if discount_distribution else 0
    avg_payment = sum(payment_distribution) / len(payment_distribution) if payment_distribution else 0

    return {
        "non_standard_contracts": sorted(non_standard, key=lambda x: x["discount_pct"], reverse=True),
        "avg_discount_pct": round(avg_discount, 1),
        "avg_payment_days": round(avg_payment, 1),
        "high_discount_count": sum(1 for d in discount_distribution if d > 20),
        "long_payment_count": sum(1 for p in payment_distribution if p > 60),
        "discount_histogram": _histogram(discount_distribution, bins=[0, 5, 10, 20, 30, 50]),
    }


def _product_performance(contracts: list) -> dict:
    """Dashboard 4: What's growing, what's slowing down?"""
    products = {}
    for c in contracts:
        products_list = _as_list(c.get("products_services", []))
        for p in products_list:
            name = p.get("name", "Unknown")
            value = p.get("value_eur", 0) or 0
            recurring = p.get("recurring", False)
            if name not in products:
                products[name] = {"total_value": 0, "arr": 0, "contracts": 0, "recurring": recurring}
            products[name]["total_value"] += value
            products[name]["contracts"] += 1
            if recurring:
                products[name]["arr"] += value

    sorted_products = sorted(products.items(), key=lambda x: x[1]["total_value"], reverse=True)

    return {
        "products": [
            {
                "name": name,
                "total_value_eur": data["total_value"],
                "arr_eur": data["arr"],
                "contract_count": data["contracts"],
                "is_recurring": data["recurring"],
                "avg_value_per_contract": round(data["total_value"] / data["contracts"], 0),
            }
            for name, data in sorted_products
        ],
        "total_products": len(products),
        "total_recurring_arr": sum(d["arr"] for _, d in sorted_products),
    }


def _risk_clauses(contracts: list) -> dict:
    """Dashboard 5: Risky clauses and total exposure."""
    all_risks = []
    total_exposure = 0

    for c in contracts:
        risk_flags = _as_list(c.get("risk_flags", []))
        penalties = _as_list(c.get("penalty_clauses", []))

        for flag in risk_flags:
            all_risks.append({
                "contract_id": c["id"],
                "client_name": c.get("client_name", "Unknown"),
                "risk_type": flag.get("flag", ""),
                "severity": flag.get("severity", "medium"),
                "clause_text": flag.get("clause", "")[:200],
            })

        for penalty in penalties:
            if isinstance(penalty, dict):
                amount = penalty.get("amount_eur", 0) or 0
                desc = penalty.get("description", str(penalty))
            else:
                amount = 0
                desc = str(penalty)
            total_exposure += amount

        # Clausola Gemella insights
        gemella_data = c.get("clausola_gemella", [])
        for gem in gemella_data:
            if gem.get("percentile_ranking", 100) < 25:
                all_risks.append({
                    "contract_id": c["id"],
                    "client_name": c.get("client_name", "Unknown"),
                    "risk_type": f"clausola_gemella_{gem.get('clause_type', 'unknown')}",
                    "severity": "high" if gem.get("percentile_ranking", 100) < 10 else "medium",
                    "clause_text": f"Bottom {100 - gem.get('percentile_ranking', 0)}% vs sector. "
                                   f"{gem.get('negotiation_script', '')[:100]}",
                })

    risk_by_severity = {
        "high": [r for r in all_risks if r["severity"] == "high"],
        "medium": [r for r in all_risks if r["severity"] == "medium"],
        "low": [r for r in all_risks if r["severity"] == "low"],
    }

    return {
        "all_risks": all_risks,
        "risk_by_severity": risk_by_severity,
        "total_penalty_exposure_eur": total_exposure,
        "high_risk_count": len(risk_by_severity["high"]),
        "contracts_with_risks": len(set(r["contract_id"] for r in all_risks)),
    }


def _portfolio_summary(contracts: list) -> dict:
    total_value = sum(c.get("total_value_eur", 0) or 0 for c in contracts)
    total_arr = sum(c.get("annual_recurring_revenue", 0) or 0 for c in contracts)
    avg_score = (
        sum(c.get("contract_score", 0) or 0 for c in contracts) / len(contracts)
        if contracts else 0
    )
    return {
        "total_contracts": len(contracts),
        "total_value_eur": total_value,
        "total_arr_eur": total_arr,
        "avg_contract_score": round(avg_score, 1),
        "computed_at": datetime.utcnow().isoformat(),
    }


def _commercial_alert(discount: float, payment_days: int) -> str:
    alerts = []
    if discount > 30:
        alerts.append(f"Sconto molto elevato ({discount}%)")
    elif discount > 20:
        alerts.append(f"Sconto sopra media ({discount}%)")
    if payment_days > 90:
        alerts.append(f"Termini di pagamento critici ({payment_days}gg)")
    elif payment_days > 60:
        alerts.append(f"Termini di pagamento allungati ({payment_days}gg)")
    return "; ".join(alerts) if alerts else "Standard"


def _histogram(values: list, bins: list) -> list:
    result = []
    for i in range(len(bins) - 1):
        count = sum(1 for v in values if bins[i] <= v < bins[i + 1])
        result.append({"range": f"{bins[i]}-{bins[i+1]}%", "count": count})
    return result


@app.get("/alerts/{client_id}")
async def get_alerts(client_id: str):
    """Pre-computed proactive alerts from the portfolio."""
    portfolio = await get_portfolio(client_id)

    critical_contracts = portfolio["dashboard_2_critical_contracts"]
    alerts = []

    for c in critical_contracts.get("expiring_contracts", []):
        if c["urgency"] in ("critical", "high"):
            alerts.append({
                "type": "expiry",
                "urgency": c["urgency"],
                "client": c["client_name"],
                "message": (
                    f"Contratto {c['client_name']} scade tra {c['days_to_expiry']} giorni. "
                    f"Valore: €{c['value_eur']:,.0f}. "
                    f"{'Rinnovo tacito: agire entro ' + c['action_deadline'] if c['auto_renewal'] and c['action_deadline'] else 'Nessun rinnovo automatico.'}"
                ),
                "revenue_at_risk_eur": c["value_eur"],
                "action": "Contattare il cliente per rinnovo",
                "deadline": c.get("action_deadline") or c["end_date"],
            })

    high_risks = portfolio["dashboard_5_risk_clauses"]["risk_by_severity"]["high"]
    for r in high_risks[:5]:
        alerts.append({
            "type": "risk_clause",
            "urgency": "high",
            "client": r["client_name"],
            "message": f"Clausola ad alto rischio in contratto {r['client_name']}: {r['clause_text'][:100]}",
            "revenue_at_risk_eur": 0,
            "action": "Revisione legale raccomandata",
            "deadline": None,
        })

    return {
        "alerts": sorted(alerts, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["urgency"]]),
        "total_alerts": len(alerts),
        "revenue_at_risk_total": sum(a["revenue_at_risk_eur"] for a in alerts),
    }
