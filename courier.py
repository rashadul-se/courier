# ============================================================================
# COURIER SERVICE ERP SYSTEM - STREAMLIT CLOUD DEPLOYMENT
# Single-file deployment with embedded FastAPI backend
# ============================================================================

# requirements.txt (create this file in your repo):
"""
streamlit==1.28.1
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
pydantic==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
requests==2.31.0
pandas==2.1.3
"""

import streamlit as st
import threading
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List
import random
import string
import enum
import re

# ============================================================================
# DATABASE SETUP (SQLite for Streamlit Cloud)
# ============================================================================

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    ForeignKey, Boolean, Enum, Text, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker

Base = declarative_base()

# Use SQLite for Streamlit Cloud (persistent storage)
import os
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./courier_erp.db")

# Create engine with SQLite
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# MODELS
# ============================================================================

class OrderStatus(enum.Enum):
    PENDING = "pending"
    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETURNED = "returned"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    orders_sent = relationship("Order", foreign_keys="Order.sender_id", back_populates="sender")
    deliveries = relationship("Delivery", back_populates="driver")

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    name = Column(String(255), nullable=False)
    phone = Column(String(20), index=True)
    email = Column(String(255), index=True)
    address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    orders = relationship("Order", back_populates="customer")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    awb_number = Column(String(50), unique=True, index=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    pickup_address = Column(Text, nullable=False)
    delivery_address = Column(Text, nullable=False)
    package_weight = Column(Float, nullable=False)
    package_dimensions = Column(String(100))
    
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, index=True)
    current_location = Column(String(255))
    
    base_price = Column(Float, nullable=False)
    cod_amount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    is_paid = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    pickup_time = Column(DateTime)
    delivered_at = Column(DateTime)
    
    customer = relationship("Customer", back_populates="orders")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="orders_sent")
    deliveries = relationship("Delivery", back_populates="order")
    tracking_events = relationship("TrackingEvent", back_populates="order", cascade="all, delete-orphan")

class Delivery(Base):
    __tablename__ = "deliveries"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    route_sequence = Column(Integer)
    estimated_distance = Column(Float)
    actual_distance = Column(Float)
    
    signature_url = Column(String(500))
    photo_url = Column(String(500))
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    delivery_notes = Column(Text)
    
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    order = relationship("Order", back_populates="deliveries")
    driver = relationship("User", back_populates="deliveries")

class TrackingEvent(Base):
    __tablename__ = "tracking_events"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    event_type = Column(String(50), nullable=False)
    description = Column(Text)
    location = Column(String(255))
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    order = relationship("Order", back_populates="tracking_events")

# ============================================================================
# SECURITY
# ============================================================================

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-" + ''.join(random.choices(string.ascii_letters, k=32)))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ============================================================================
# BUSINESS LOGIC SERVICES
# ============================================================================

class OrderService:
    @staticmethod
    def generate_awb() -> str:
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        random_suffix = ''.join(random.choices(string.digits, k=6))
        return f"AWB{timestamp}{random_suffix}"
    
    @staticmethod
    def calculate_price(weight: float, distance: float = 10.0) -> float:
        base_rate = 5.0
        weight_rate = 2.0 * weight
        distance_rate = 0.5 * distance
        return round(base_rate + weight_rate + distance_rate, 2)
    
    @classmethod
    def create_order(cls, db: Session, customer_id: int, pickup_address: str, 
                     delivery_address: str, package_weight: float, 
                     cod_amount: float = 0.0, package_dimensions: str = None) -> Order:
        awb = cls.generate_awb()
        base_price = cls.calculate_price(package_weight)
        total = base_price + cod_amount
        
        new_order = Order(
            awb_number=awb,
            customer_id=customer_id,
            pickup_address=pickup_address,
            delivery_address=delivery_address,
            package_weight=package_weight,
            package_dimensions=package_dimensions,
            base_price=base_price,
            cod_amount=cod_amount,
            total_amount=total,
            status=OrderStatus.PENDING
        )
        
        db.add(new_order)
        db.commit()
        db.refresh(new_order)
        
        TrackingService.add_event(
            db, new_order.id, "ORDER_CREATED", 
            f"Order created with AWB: {awb}"
        )
        
        return new_order
    
    @staticmethod
    def update_status(db: Session, order_id: int, new_status: OrderStatus, location: str = None):
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            raise ValueError("Order not found")
        
        order.status = new_status
        if location:
            order.current_location = location
        
        if new_status == OrderStatus.DELIVERED:
            order.delivered_at = datetime.utcnow()
        
        db.commit()
        
        TrackingService.add_event(
            db, order_id, new_status.value.upper(), 
            f"Status updated to {new_status.value}", location
        )

class TrackingService:
    @staticmethod
    def add_event(db: Session, order_id: int, event_type: str, 
                  description: str, location: str = None):
        event = TrackingEvent(
            order_id=order_id,
            event_type=event_type,
            description=description,
            location=location,
            created_at=datetime.utcnow()
        )
        db.add(event)
        db.commit()
    
    @staticmethod
    def get_tracking_info(db: Session, awb_number: str) -> dict:
        order = db.query(Order).filter(Order.awb_number == awb_number).first()
        if not order:
            raise ValueError("Order not found")
        
        events = db.query(TrackingEvent)\
            .filter(TrackingEvent.order_id == order.id)\
            .order_by(TrackingEvent.created_at.desc())\
            .all()
        
        return {
            "order_id": order.id,
            "awb_number": order.awb_number,
            "status": order.status.value,
            "current_location": order.current_location,
            "events": [
                {
                    "type": e.event_type,
                    "description": e.description,
                    "location": e.location,
                    "timestamp": e.created_at.isoformat()
                } for e in events
            ]
        }

# ============================================================================
# FASTAPI BACKEND
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Courier ERP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class OrderCreate(BaseModel):
    customer_id: int
    pickup_address: str = Field(..., min_length=10)
    delivery_address: str = Field(..., min_length=10)
    package_weight: float = Field(..., gt=0, le=1000)
    cod_amount: float = Field(default=0.0, ge=0)
    package_dimensions: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/orders")
async def create_order(order: OrderCreate):
    db = next(get_db())
    try:
        new_order = OrderService.create_order(
            db, order.customer_id, order.pickup_address,
            order.delivery_address, order.package_weight,
            order.cod_amount, order.package_dimensions
        )
        return {
            "id": new_order.id,
            "awb_number": new_order.awb_number,
            "status": new_order.status.value,
            "total_amount": new_order.total_amount,
            "created_at": new_order.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/orders/{order_id}")
async def get_order(order_id: int):
    db = next(get_db())
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {
        "id": order.id,
        "awb_number": order.awb_number,
        "status": order.status.value,
        "total_amount": order.total_amount,
        "created_at": order.created_at.isoformat()
    }

@app.get("/api/tracking/{awb_number}")
async def track_order(awb_number: str):
    db = next(get_db())
    try:
        return TrackingService.get_tracking_info(db, awb_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/orders/{order_id}/status")
async def update_order_status(order_id: int, status: str, location: Optional[str] = None):
    db = next(get_db())
    try:
        order_status = OrderStatus[status.upper()]
        OrderService.update_status(db, order_id, order_status, location)
        return {"message": "Status updated successfully"}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid status")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/reports/dashboard")
async def get_dashboard_stats():
    from sqlalchemy import func
    db = next(get_db())
    
    total_orders = db.query(func.count(Order.id)).scalar() or 0
    delivered_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar() or 0
    
    revenue = db.query(func.sum(Order.total_amount))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar() or 0
    
    pending_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.PENDING).scalar() or 0
    
    return {
        "total_orders": total_orders,
        "delivered_orders": delivered_orders,
        "delivery_rate": round(delivered_orders / total_orders * 100, 2) if total_orders > 0 else 0,
        "total_revenue": round(revenue, 2),
        "pending_orders": pending_orders
    }

@app.get("/api/orders")
async def list_orders(skip: int = 0, limit: int = 20):
    db = next(get_db())
    orders = db.query(Order).order_by(Order.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": o.id,
            "awb_number": o.awb_number,
            "status": o.status.value,
            "total_amount": o.total_amount,
            "created_at": o.created_at.isoformat()
        } for o in orders
    ]

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database with tables and sample data"""
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    # Check if we already have data
    existing_customers = db.query(Customer).count()
    if existing_customers == 0:
        # Create sample customer
        sample_customer = Customer(
            name="John Doe",
            phone="+1234567890",
            email="john@example.com",
            address="123 Main Street, New York, NY 10001"
        )
        db.add(sample_customer)
        
        # Create sample user
        sample_user = User(
            email="admin@courier.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Admin User",
            role="admin",
            is_active=True
        )
        db.add(sample_user)
        
        db.commit()
        print("✅ Database initialized with sample data")
    
    db.close()

# ============================================================================
# START FASTAPI IN BACKGROUND THREAD
# ============================================================================

def run_fastapi():
    """Run FastAPI in background thread"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

def start_backend():
    """Start FastAPI backend if not already running"""
    if 'backend_started' not in st.session_state:
        init_database()  # Initialize DB first
        backend_thread = threading.Thread(target=run_fastapi, daemon=True)
        backend_thread.start()
        time.sleep(2)  # Wait for server to start
        st.session_state.backend_started = True
        print("✅ FastAPI backend started on port 8000")

# ============================================================================
# STREAMLIT FRONTEND
# ============================================================================

# Start backend
start_backend()

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Courier ERP System", page_icon="📦", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .success-box {
        padding: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📦 Courier ERP")
st.sidebar.markdown("---")
menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "📝 Create Order", "🔍 Track Shipment", "📋 Orders List", "📊 Reports"]
)

# Dashboard
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Overview")
    
    try:
        response = requests.get(f"{API_BASE_URL}/reports/dashboard", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📦 Total Orders", data['total_orders'])
            with col2:
                st.metric("✅ Delivered", data['delivered_orders'])
            with col3:
                st.metric("📈 Delivery Rate", f"{data['delivery_rate']}%")
            with col4:
                st.metric("💰 Revenue", f"${data['total_revenue']:,.2f}")
            
            st.markdown("---")
            
            # Recent orders
            st.subheader("📋 Recent Orders")
            orders_response = requests.get(f"{API_BASE_URL}/orders?limit=10", timeout=5)
            if orders_response.status_code == 200:
                orders = orders_response.json()
                if orders:
                    df = pd.DataFrame(orders)
                    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No orders yet. Create your first order!")
        else:
            st.error("❌ Failed to load dashboard")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure the backend is running")

# Create Order
elif menu == "📝 Create Order":
    st.title("📝 Create New Order")
    
    with st.form("order_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, value=1, step=1)
            pickup_address = st.text_area("Pickup Address", height=100, 
                                         placeholder="Enter pickup address (min 10 characters)")
            package_weight = st.number_input("Package Weight (kg)", min_value=0.1, value=1.0, step=0.1)
        
        with col2:
            delivery_address = st.text_area("Delivery Address", height=100,
                                           placeholder="Enter delivery address (min 10 characters)")
            cod_amount = st.number_input("COD Amount ($)", min_value=0.0, value=0.0, step=0.01)
            package_dimensions = st.text_input("Dimensions (optional)", placeholder="L x W x H cm")
        
        submitted = st.form_submit_button("🚀 Create Order", use_container_width=True)
        
        if submitted:
            if len(pickup_address) < 10 or len(delivery_address) < 10:
                st.error("❌ Addresses must be at least 10 characters long")
            else:
                order_data = {
                    "customer_id": customer_id,
                    "pickup_address": pickup_address,
                    "delivery_address": delivery_address,
                    "package_weight": package_weight,
                    "cod_amount": cod_amount,
                    "package_dimensions": package_dimensions if package_dimensions else None
                }
                
                try:
                    response = requests.post(f"{API_BASE_URL}/orders", json=order_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Order created successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AWB Number", result['awb_number'])
                        with col2:
                            st.metric("Total Amount", f"${result['total_amount']}")
                        with col3:
                            st.metric("Status", result['status'].upper())
                        
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Track Shipment
elif menu == "🔍 Track Shipment":
    st.title("🔍 Track Your Shipment")
    
    awb_number = st.text_input("Enter AWB Number", placeholder="AWB20241115...", key="track_awb")
    
    if st.button("🔍 Track", use_container_width=True):
        if awb_number:
            try:
                response = requests.get(f"{API_BASE_URL}/tracking/{awb_number}", timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"✅ Found! Status: **{data['status'].upper()}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("AWB Number", data['awb_number'])
                    with col2:
                        st.metric("Current Location", data['current_location'] or "N/A")
                    
                    st.markdown("---")
                    st.subheader("📍 Tracking History")
                    
                    for event in data['events']:
                        with st.expander(f"🚚 {event['type']} - {event['timestamp'][:19]}", expanded=True):
                            st.write(f"**Description:** {event['description']}")
                            if event['location']:
                                st.write(f"**Location:** 📍 {event['location']}")
                else:
                    st.error("❌ Order not found. Please check the AWB number.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter an AWB number")

# Orders List
elif menu == "📋 Orders List":
    st.title("📋 All Orders")
    
    try:
        response = requests.get(f"{API_BASE_URL}/orders?limit=50", timeout=5)
        
        if response.status_code == 200:
            orders = response.json()
            
            if orders:
                df = pd.DataFrame(orders)
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.multiselect("Filter by Status", 
                                                  options=df['status'].unique(),
                                                  default=df['status'].unique())
                
                filtered_df = df[df['status'].isin(status_filter)]
                
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                st.info(f"📊 Showing {len(filtered_df)} orders")
            else:
                st.info("📭 No orders yet. Create your first order!")
        else:
            st.error("❌ Failed to load orders")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Reports
elif menu == "📊 Reports":
    st.title("📊 Reports & Analytics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/reports/dashboard", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            st.subheader("📈 Performance Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Orders Processed", data['total_orders'])
                st.metric("Pending Orders", data['pending_orders'])
            with col2:
                st.metric("Successful Deliveries", data['delivered_orders'])
                st.metric("Total Revenue Generated", f"${data['total_revenue']:,.2f}")
            
            # Progress bar
            st.markdown("---")
            st.subheader("🎯 Delivery Performance")
            delivery_rate = data['delivery_rate']
            st.progress(delivery_rate / 100)
            st.write(f"**On-Time Delivery Rate:** {delivery_rate}%")
            
            if delivery_rate >= 90:
                st.success("🎉 Excellent performance!")
            elif delivery_rate >= 70:
                st.info("👍 Good performance")
            else:
                st.warning("⚠️ Needs improvement")
        else:
            st.error("❌ Failed to load reports")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Courier ERP System v1.0**

Features:
- ✅ Order Management
- ✅ Real-time Tracking
- ✅ Automated Invoicing
- ✅ Dashboard Analytics
""")

st.sidebar.success("✅ Backend Running")
st.sidebar.caption(f"Database: SQLite")
st.sidebar.caption(f"API: localhost:8000")
