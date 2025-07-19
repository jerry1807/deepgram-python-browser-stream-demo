import asyncio
import json
from datetime import datetime, timedelta
import random
import pathlib

# Hardcoded user context
HARDCODED_USER = {
    "username": "johndoe",
    "phone": "+15551234567",
    "email": "john.doe@example.com",
    "name": "John Doe",
    "reward_points": 1200,
    "customer_id": "CUST0001"
}

HARDCODED_APPOINTMENTS = [
    {
        "id": "APT0001",
        "customer_id": "CUST0001",
        "customer_name": "John Doe",
        "date": "2024-07-20T14:00:00",
        "service": "Consultation",
        "status": "Scheduled"
    }
]

HARDCODED_ORDERS = [
    {
        "id": "ORD0001",
        "customer_id": "CUST0001",
        "items": 3,
        "total": "$59.99",
        "status": "Shipped",
        "date": "2024-07-10"
    }
]

async def get_customer(phone=None, email=None, customer_id=None):
    """Look up a customer by phone, email, or ID."""
    # Always return the hardcoded user
    return HARDCODED_USER

async def get_customer_appointments(customer_id):
    """Get all appointments for a customer."""
    return {"customer_id": HARDCODED_USER["customer_id"], "appointments": HARDCODED_APPOINTMENTS}

async def get_customer_orders(customer_id):
    """Get all orders for a customer."""
    return {"customer_id": HARDCODED_USER["customer_id"], "orders": HARDCODED_ORDERS}

async def schedule_appointment(customer_id, date, service):
    """Schedule a new appointment."""
    # Just return a fake appointment confirmation
    return {
        "id": "APT0002",
        "customer_id": HARDCODED_USER["customer_id"],
        "customer_name": HARDCODED_USER["name"],
        "date": date,
        "service": service,
        "status": "Scheduled"
    }

async def get_available_appointment_slots(start_date, end_date):
    """Get available appointment slots."""
    # Return a couple of hardcoded slots
    return {"available_slots": [
        "2024-07-21T10:00:00",
        "2024-07-21T11:00:00"
    ]}

async def prepare_agent_filler_message(websocket, message_type):
    result = {"status": "queued", "message_type": message_type}
    if message_type == "lookup":
        inject_message = {"type": "InjectAgentMessage", "message": "Let me look that up for you..."}
    else:
        inject_message = {"type": "InjectAgentMessage", "message": "One moment please..."}
    return {"function_response": result, "inject_message": inject_message}

async def prepare_farewell_message(websocket, farewell_type):
    if farewell_type == "thanks":
        message = "Thank you for calling! Have a great day!"
    elif farewell_type == "help":
        message = "I'm glad I could help! Have a wonderful day!"
    else:
        message = "Goodbye! Have a nice day!"
    inject_message = {"type": "InjectAgentMessage", "message": message}
    close_message = {"type": "close"}
    return {"function_response": {"status": "closing", "message": message}, "inject_message": inject_message, "close_message": close_message}
