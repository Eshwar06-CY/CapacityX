"""KIT EXAMPLE tools: an online store's order system, with in-memory data so the demo runs anywhere.

Replace with your own tools (/new-agent does this). Keep a clear one-line docstring on each: the
model reads it, and the tool-selection pattern uses it as the option description.
"""

from __future__ import annotations

from strands import tool

ORDERS = {
    "A1001": {"item": "trail running shoes", "amount": 129.00, "status": "delivered", "customer": "Priya"},
    "A1002": {"item": "noise-cancelling headphones", "amount": 249.00, "status": "shipped", "customer": "Arun"},
    "A1003": {"item": "yoga mat", "amount": 35.00, "status": "processing", "customer": "Meena"},
}
ACTIONS: list[dict] = []


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order's item, amount and status by its order id."""
    o = ORDERS.get(order_id.strip().upper())
    return {"order_id": order_id, **o} if o else {"error": f"no order {order_id}"}


@tool
def refund_order(order_id: str, amount: float) -> dict:
    """Refund an order (irreversible): sends the money back to the customer."""
    o = ORDERS.get(order_id.strip().upper())
    if not o:
        return {"error": f"no order {order_id}"}
    if amount > o["amount"]:
        return {"error": f"refund {amount} exceeds the order amount {o['amount']}"}
    ACTIONS.append({"action": "refund", "order_id": order_id, "amount": amount})
    return {"refunded": amount, "order_id": order_id}


@tool
def cancel_order(order_id: str) -> dict:
    """Cancel an order that has not shipped yet (irreversible)."""
    o = ORDERS.get(order_id.strip().upper())
    if not o:
        return {"error": f"no order {order_id}"}
    if o["status"] != "processing":
        return {"error": f"order {order_id} is {o['status']} and cannot be cancelled"}
    o["status"] = "cancelled"
    ACTIONS.append({"action": "cancel", "order_id": order_id})
    return {"cancelled": order_id}


TOOLS = [lookup_order, refund_order, cancel_order]
