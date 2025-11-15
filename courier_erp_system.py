# ============================================================================
# COURIER SERVICE ERP SYSTEM - COMPLETE IMPLEMENTATION
# Architecture: Microservices + FastAPI + Streamlit + PostgreSQL
# ============================================================================

# requirements.txt:
# fastapi==0.104.1
# uvicorn[standard]==0.24.0
# streamlit==1.28.1
# sqlalchemy==2.0.23
# asyncpg==0.29.0
# pydantic==2.5.0
# pydantic[email]==2.5.0
# email-validator==2.1.0
# pydantic-settings==2.1.0
# python-jose[cryptography]==3.3.0
# passlib[bcrypt]==1.7.4
# python-multipart==0.0.6
# redis==5.0.1
# celery==5.3.4

# ============================================================================
# 1. DATABASE MODELS WITH OPTIMIZED INDEXING
# ============================================================================

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    ForeignKey, Boolean, Enum, Text, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from datetime import datetime
import enum

Base = declarative_base()

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
    role = Column(String(50), nullable=False)  # admin, manager, driver, customer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    orders_sent = relationship("Order", foreign_keys="Order.sender_id", back_populates="sender")
    deliveries = relationship("Delivery", back_populates="driver")
    
    __table_args__ = (
        Index('idx_user_role_active', 'role', 'is_active'),
    )

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
    
    # Shipment details
    pickup_address = Column(Text, nullable=False)
    delivery_address = Column(Text, nullable=False)
    package_weight = Column(Float, nullable=False)
    package_dimensions = Column(String(100))
    
    # Status and tracking
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, index=True)
    current_location = Column(String(255))
    
    # Financial
    base_price = Column(Float, nullable=False)
    cod_amount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    is_paid = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    pickup_time = Column(DateTime)
    delivered_at = Column(DateTime)
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="orders_sent")
    deliveries = relationship("Delivery", back_populates="order")
    tracking_events = relationship("TrackingEvent", back_populates="order", cascade="all, delete-orphan")
    invoice = relationship("Invoice", back_populates="order", uselist=False)
    
    __table_args__ = (
        Index('idx_order_status_date', 'status', 'created_at'),
        Index('idx_order_customer_status', 'customer_id', 'status'),
        CheckConstraint('package_weight > 0', name='check_positive_weight'),
        CheckConstraint('total_amount >= 0', name='check_positive_amount'),
    )

class Delivery(Base):
    __tablename__ = "deliveries"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    
    # Route information
    route_sequence = Column(Integer)
    estimated_distance = Column(Float)
    actual_distance = Column(Float)
    
    # Delivery proof
    signature_url = Column(String(500))
    photo_url = Column(String(500))
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    delivery_notes = Column(Text)
    
    # Timestamps
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime, index=True)
    
    # Relationships
    order = relationship("Order", back_populates="deliveries")
    driver = relationship("User", back_populates="deliveries")
    vehicle = relationship("Vehicle", back_populates="deliveries")
    
    __table_args__ = (
        Index('idx_delivery_driver_date', 'driver_id', 'assigned_at'),
    )

class Vehicle(Base):
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(50), unique=True, index=True)
    vehicle_type = Column(String(50))
    capacity_kg = Column(Float)
    fuel_type = Column(String(20))
    
    # Maintenance
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    current_mileage = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    current_driver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    deliveries = relationship("Delivery", back_populates="vehicle")

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
    
    __table_args__ = (
        Index('idx_tracking_order_date', 'order_id', 'created_at'),
    )

class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String(50), unique=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    
    # Financial details
    subtotal = Column(Float, nullable=False)
    tax_amount = Column(Float, default=0.0)
    discount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    
    # Payment
    payment_status = Column(String(20), default="pending", index=True)
    payment_method = Column(String(50))
    paid_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    order = relationship("Order", back_populates="invoice")
    
    __table_args__ = (
        Index('idx_invoice_status_date', 'payment_status', 'created_at'),
    )

class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    employee_code = Column(String(50), unique=True, index=True)
    department = Column(String(100))
    designation = Column(String(100))
    
    # Salary
    base_salary = Column(Float)
    bonus_rate = Column(Float, default=0.0)
    
    # Performance
    on_time_delivery_rate = Column(Float, default=0.0)
    total_deliveries = Column(Integer, default=0)
    
    # Compliance
    license_number = Column(String(50))
    license_expiry = Column(DateTime)
    
    joined_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# ============================================================================
# 2. PYDANTIC SCHEMAS FOR VALIDATION
# ============================================================================

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime
import re

class OrderCreate(BaseModel):
    customer_id: int
    pickup_address: str = Field(..., min_length=10, max_length=500)
    delivery_address: str = Field(..., min_length=10, max_length=500)
    package_weight: float = Field(..., gt=0, le=1000)
    package_dimensions: Optional[str] = None
    cod_amount: float = Field(default=0.0, ge=0)
    
    @field_validator('pickup_address', 'delivery_address')
    @classmethod
    def validate_address(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Address must be at least 10 characters')
        return v.strip()

class OrderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    awb_number: str
    status: str
    total_amount: float
    created_at: datetime

class TrackingResponse(BaseModel):
    order_id: int
    awb_number: str
    status: str
    current_location: Optional[str]
    events: List[dict]

class UserCreate(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=255)
    role: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        # Simple email validation without external dependency
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}


# ============================================================================
# 3. DATABASE CONNECTION & OPTIMIZATION
# ============================================================================

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/courier_erp"

# Connection pooling for performance
engine = create_engine(
    DATABASE_URL.replace('+asyncpg', ''),
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# 4. SECURITY & AUTHENTICATION
# ============================================================================

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# ============================================================================
# 5. BUSINESS LOGIC SERVICES (OOP)
# ============================================================================

import random
import string
from sqlalchemy.orm import Session

class OrderService:
    """Handles all order-related business logic"""
    
    @staticmethod
    def generate_awb() -> str:
        """Generate unique Airway Bill number"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        random_suffix = ''.join(random.choices(string.digits, k=6))
        return f"AWB{timestamp}{random_suffix}"
    
    @staticmethod
    def calculate_price(weight: float, distance: float = 10.0) -> float:
        """Calculate shipping price based on weight and distance"""
        base_rate = 5.0
        weight_rate = 2.0 * weight
        distance_rate = 0.5 * distance
        return round(base_rate + weight_rate + distance_rate, 2)
    
    @classmethod
    def create_order(cls, db: Session, order_data: OrderCreate) -> Order:
        """Create new order with automatic pricing"""
        awb = cls.generate_awb()
        base_price = cls.calculate_price(order_data.package_weight)
        total = base_price + order_data.cod_amount
        
        new_order = Order(
            awb_number=awb,
            customer_id=order_data.customer_id,
            pickup_address=order_data.pickup_address,
            delivery_address=order_data.delivery_address,
            package_weight=order_data.package_weight,
            package_dimensions=order_data.package_dimensions,
            base_price=base_price,
            cod_amount=order_data.cod_amount,
            total_amount=total,
            status=OrderStatus.PENDING
        )
        
        db.add(new_order)
        db.commit()
        db.refresh(new_order)
        
        # Create tracking event
        TrackingService.add_event(
            db, new_order.id, "ORDER_CREATED", 
            f"Order created with AWB: {awb}"
        )
        
        return new_order
    
    @staticmethod
    def update_status(db: Session, order_id: int, new_status: OrderStatus, location: str = None):
        """Update order status and create tracking event"""
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
    """Manages shipment tracking and events"""
    
    @staticmethod
    def add_event(db: Session, order_id: int, event_type: str, 
                  description: str, location: str = None):
        """Add tracking event for an order"""
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
        """Get complete tracking information"""
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

class InvoiceService:
    """Handles billing and invoicing"""
    
    @staticmethod
    def generate_invoice(db: Session, order_id: int) -> Invoice:
        """Generate invoice for completed order"""
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            raise ValueError("Order not found")
        
        # Check if invoice already exists
        existing = db.query(Invoice).filter(Invoice.order_id == order_id).first()
        if existing:
            return existing
        
        invoice_number = f"INV{datetime.utcnow().strftime('%Y%m%d')}{order.id:06d}"
        tax_rate = 0.18  # 18% tax
        
        subtotal = order.base_price
        tax = round(subtotal * tax_rate, 2)
        total = subtotal + tax + order.cod_amount
        
        invoice = Invoice(
            invoice_number=invoice_number,
            order_id=order_id,
            subtotal=subtotal,
            tax_amount=tax,
            total_amount=total,
            payment_status="pending"
        )
        
        db.add(invoice)
        db.commit()
        db.refresh(invoice)
        
        return invoice

class RouteOptimizationService:
    """Optimizes delivery routes (simplified version)"""
    
    @staticmethod
    def assign_driver(db: Session, order_id: int, driver_id: int):
        """Assign driver to order"""
        delivery = Delivery(
            order_id=order_id,
            driver_id=driver_id,
            assigned_at=datetime.utcnow()
        )
        db.add(delivery)
        db.commit()
        
        OrderService.update_status(
            db, order_id, OrderStatus.PICKED_UP, 
            "Assigned to driver"
        )


# ============================================================================
# 6. FASTAPI MICROSERVICE - ORDER SERVICE
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(title="Courier ERP - Order Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Login endpoint with dummy users"""
    # Dummy users for testing
    DUMMY_USERS = {
        "admin@courier.com": {
            "password": "admin123",
            "full_name": "Admin User",
            "role": "admin",
            "id": 1
        },
        "manager@courier.com": {
            "password": "manager123",
            "full_name": "Manager User",
            "role": "manager",
            "id": 2
        },
        "driver@courier.com": {
            "password": "driver123",
            "full_name": "Driver User",
            "role": "driver",
            "id": 3
        },
        "customer@courier.com": {
            "password": "customer123",
            "full_name": "Customer User",
            "role": "customer",
            "id": 4
        }
    }
    
    # Check dummy users first
    if login_data.email in DUMMY_USERS:
        dummy_user = DUMMY_USERS[login_data.email]
        if login_data.password == dummy_user["password"]:
            # Create access token
            access_token = create_access_token(
                data={"sub": login_data.email, "role": dummy_user["role"]},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": dummy_user["id"],
                    "email": login_data.email,
                    "full_name": dummy_user["full_name"],
                    "role": dummy_user["role"]
                }
            }
    
    # Check database for real users
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": "User registered successfully",
        "user_id": new_user.id,
        "email": new_user.email
    }

@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current logged-in user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active
    }

@app.post("/api/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order: OrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create new shipping order"""
    try:
        new_order = OrderService.create_order(db, order)
        return OrderResponse.from_orm(new_order)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get order details"""
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse.from_orm(order)

@app.get("/api/tracking/{awb_number}", response_model=TrackingResponse)
async def track_order(awb_number: str, db: Session = Depends(get_db)):
    """Public tracking endpoint - no authentication required"""
    try:
        tracking_info = TrackingService.get_tracking_info(db, awb_number)
        return tracking_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    status: str,
    location: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update order status"""
    try:
        order_status = OrderStatus[status.upper()]
        OrderService.update_status(db, order_id, order_status, location)
        return {"message": "Status updated successfully"}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid status")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/orders/{order_id}/invoice")
async def generate_order_invoice(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate invoice for order"""
    try:
        invoice = InvoiceService.generate_invoice(db, order_id)
        return {
            "invoice_number": invoice.invoice_number,
            "total_amount": invoice.total_amount,
            "payment_status": invoice.payment_status
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/reports/dashboard")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get dashboard statistics with optimized queries"""
    from sqlalchemy import func
    
    # Optimized aggregation queries
    total_orders = db.query(func.count(Order.id)).scalar()
    delivered_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar()
    
    revenue = db.query(func.sum(Order.total_amount))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar() or 0
    
    pending_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.PENDING).scalar()
    
    return {
        "total_orders": total_orders,
        "delivered_orders": delivered_orders,
        "delivery_rate": round(delivered_orders / total_orders * 100, 2) if total_orders > 0 else 0,
        "total_revenue": round(revenue, 2),
        "pending_orders": pending_orders
    }


# ============================================================================
# 7. STREAMLIT FRONTEND
# ============================================================================

"""
Save this as streamlit_app.py and run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Courier ERP System", page_icon="📦", layout="wide")

# Session state for authentication
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

def get_headers():
    """Get authorization headers"""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

# Sidebar navigation
st.sidebar.title("📦 Courier ERP")
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Create Order", "Track Shipment", "Orders List", "Reports"]
)

# Dashboard
if menu == "Dashboard":
    st.title("📊 Dashboard")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/reports/dashboard",
            headers=get_headers()
        )
        
        if response.status_code == 200:
            data = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Orders", data['total_orders'])
            with col2:
                st.metric("Delivered", data['delivered_orders'])
            with col3:
                st.metric("Delivery Rate", f"{data['delivery_rate']}%")
            with col4:
                st.metric("Revenue", f"${data['total_revenue']:,.2f}")
            
            st.success("Dashboard loaded successfully")
        else:
            st.error("Failed to load dashboard")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Create Order
elif menu == "Create Order":
    st.title("📝 Create New Order")
    
    with st.form("order_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, step=1)
            pickup_address = st.text_area("Pickup Address", height=100)
            package_weight = st.number_input("Package Weight (kg)", min_value=0.1, step=0.1)
        
        with col2:
            delivery_address = st.text_area("Delivery Address", height=100)
            cod_amount = st.number_input("COD Amount ($)", min_value=0.0, step=0.01)
            package_dimensions = st.text_input("Dimensions (optional)")
        
        submitted = st.form_submit_button("Create Order")
        
        if submitted:
            order_data = {
                "customer_id": customer_id,
                "pickup_address": pickup_address,
                "delivery_address": delivery_address,
                "package_weight": package_weight,
                "cod_amount": cod_amount,
                "package_dimensions": package_dimensions
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/orders",
                    json=order_data,
                    headers=get_headers()
                )
                
                if response.status_code == 201:
                    result = response.json()
                    st.success(f"✅ Order created successfully!")
                    st.info(f"AWB Number: {result['awb_number']}")
                    st.info(f"Total Amount: ${result['total_amount']}")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Track Shipment
elif menu == "Track Shipment":
    st.title("🔍 Track Your Shipment")
    
    awb_number = st.text_input("Enter AWB Number", placeholder="AWB20241115...")
    
    if st.button("Track"):
        try:
            response = requests.get(f"{API_BASE_URL}/tracking/{awb_number}")
            
            if response.status_code == 200:
                data = response.json()
                
                st.success(f"Status: **{data['status'].upper()}**")
                if data['current_location']:
                    st.info(f"📍 Current Location: {data['current_location']}")
                
                st.subheader("Tracking History")
                for event in data['events']:
                    with st.expander(f"{event['type']} - {event['timestamp']}"):
                        st.write(f"**Description:** {event['description']}")
                        if event['location']:
                            st.write(f"**Location:** {event['location']}")
            else:
                st.error("Order not found")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Orders List
elif menu == "Orders List":
    st.title("📋 All Orders")
    st.info("This would display a paginated list of all orders with filters")

# Reports
elif menu == "Reports":
    st.title("📈 Reports & Analytics")
    st.info("Advanced reporting features would be implemented here")


# ============================================================================
# 8. INITIALIZE DATABASE
# ============================================================================

def init_database():
    """Initialize database with tables and indexes"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_database()
    print("Starting FastAPI server...")
    print("Run with: uvicorn main:app --reload")
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ['admin', 'manager', 'driver', 'customer']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of {allowed_roles}')
        return v


# ============================================================================
# 3. DATABASE CONNECTION & OPTIMIZATION
# ============================================================================

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/courier_erp"

# Connection pooling for performance
engine = create_engine(
    DATABASE_URL.replace('+asyncpg', ''),
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# 4. SECURITY & AUTHENTICATION
# ============================================================================

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# ============================================================================
# 5. BUSINESS LOGIC SERVICES (OOP)
# ============================================================================

import random
import string
from sqlalchemy.orm import Session

class OrderService:
    """Handles all order-related business logic"""
    
    @staticmethod
    def generate_awb() -> str:
        """Generate unique Airway Bill number"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        random_suffix = ''.join(random.choices(string.digits, k=6))
        return f"AWB{timestamp}{random_suffix}"
    
    @staticmethod
    def calculate_price(weight: float, distance: float = 10.0) -> float:
        """Calculate shipping price based on weight and distance"""
        base_rate = 5.0
        weight_rate = 2.0 * weight
        distance_rate = 0.5 * distance
        return round(base_rate + weight_rate + distance_rate, 2)
    
    @classmethod
    def create_order(cls, db: Session, order_data: OrderCreate) -> Order:
        """Create new order with automatic pricing"""
        awb = cls.generate_awb()
        base_price = cls.calculate_price(order_data.package_weight)
        total = base_price + order_data.cod_amount
        
        new_order = Order(
            awb_number=awb,
            customer_id=order_data.customer_id,
            pickup_address=order_data.pickup_address,
            delivery_address=order_data.delivery_address,
            package_weight=order_data.package_weight,
            package_dimensions=order_data.package_dimensions,
            base_price=base_price,
            cod_amount=order_data.cod_amount,
            total_amount=total,
            status=OrderStatus.PENDING
        )
        
        db.add(new_order)
        db.commit()
        db.refresh(new_order)
        
        # Create tracking event
        TrackingService.add_event(
            db, new_order.id, "ORDER_CREATED", 
            f"Order created with AWB: {awb}"
        )
        
        return new_order
    
    @staticmethod
    def update_status(db: Session, order_id: int, new_status: OrderStatus, location: str = None):
        """Update order status and create tracking event"""
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
    """Manages shipment tracking and events"""
    
    @staticmethod
    def add_event(db: Session, order_id: int, event_type: str, 
                  description: str, location: str = None):
        """Add tracking event for an order"""
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
        """Get complete tracking information"""
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

class InvoiceService:
    """Handles billing and invoicing"""
    
    @staticmethod
    def generate_invoice(db: Session, order_id: int) -> Invoice:
        """Generate invoice for completed order"""
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            raise ValueError("Order not found")
        
        # Check if invoice already exists
        existing = db.query(Invoice).filter(Invoice.order_id == order_id).first()
        if existing:
            return existing
        
        invoice_number = f"INV{datetime.utcnow().strftime('%Y%m%d')}{order.id:06d}"
        tax_rate = 0.18  # 18% tax
        
        subtotal = order.base_price
        tax = round(subtotal * tax_rate, 2)
        total = subtotal + tax + order.cod_amount
        
        invoice = Invoice(
            invoice_number=invoice_number,
            order_id=order_id,
            subtotal=subtotal,
            tax_amount=tax,
            total_amount=total,
            payment_status="pending"
        )
        
        db.add(invoice)
        db.commit()
        db.refresh(invoice)
        
        return invoice

class RouteOptimizationService:
    """Optimizes delivery routes (simplified version)"""
    
    @staticmethod
    def assign_driver(db: Session, order_id: int, driver_id: int):
        """Assign driver to order"""
        delivery = Delivery(
            order_id=order_id,
            driver_id=driver_id,
            assigned_at=datetime.utcnow()
        )
        db.add(delivery)
        db.commit()
        
        OrderService.update_status(
            db, order_id, OrderStatus.PICKED_UP, 
            "Assigned to driver"
        )


# ============================================================================
# 6. FASTAPI MICROSERVICE - ORDER SERVICE
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(title="Courier ERP - Order Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order: OrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create new shipping order"""
    try:
        new_order = OrderService.create_order(db, order)
        return OrderResponse.from_orm(new_order)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get order details"""
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse.from_orm(order)

@app.get("/api/tracking/{awb_number}", response_model=TrackingResponse)
async def track_order(awb_number: str, db: Session = Depends(get_db)):
    """Public tracking endpoint - no authentication required"""
    try:
        tracking_info = TrackingService.get_tracking_info(db, awb_number)
        return tracking_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    status: str,
    location: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update order status"""
    try:
        order_status = OrderStatus[status.upper()]
        OrderService.update_status(db, order_id, order_status, location)
        return {"message": "Status updated successfully"}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid status")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/orders/{order_id}/invoice")
async def generate_order_invoice(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate invoice for order"""
    try:
        invoice = InvoiceService.generate_invoice(db, order_id)
        return {
            "invoice_number": invoice.invoice_number,
            "total_amount": invoice.total_amount,
            "payment_status": invoice.payment_status
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/reports/dashboard")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get dashboard statistics with optimized queries"""
    from sqlalchemy import func
    
    # Optimized aggregation queries
    total_orders = db.query(func.count(Order.id)).scalar()
    delivered_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar()
    
    revenue = db.query(func.sum(Order.total_amount))\
        .filter(Order.status == OrderStatus.DELIVERED).scalar() or 0
    
    pending_orders = db.query(func.count(Order.id))\
        .filter(Order.status == OrderStatus.PENDING).scalar()
    
    return {
        "total_orders": total_orders,
        "delivered_orders": delivered_orders,
        "delivery_rate": round(delivered_orders / total_orders * 100, 2) if total_orders > 0 else 0,
        "total_revenue": round(revenue, 2),
        "pending_orders": pending_orders
    }


# ============================================================================
# 7. STREAMLIT FRONTEND
# ============================================================================

"""
Save this as streamlit_app.py and run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Courier ERP System", page_icon="📦", layout="wide")

# Session state for authentication
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

def get_headers():
    """Get authorization headers"""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

# Sidebar navigation
st.sidebar.title("📦 Courier ERP")
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Create Order", "Track Shipment", "Orders List", "Reports"]
)

# Dashboard
if menu == "Dashboard":
    st.title("📊 Dashboard")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/reports/dashboard",
            headers=get_headers()
        )
        
        if response.status_code == 200:
            data = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Orders", data['total_orders'])
            with col2:
                st.metric("Delivered", data['delivered_orders'])
            with col3:
                st.metric("Delivery Rate", f"{data['delivery_rate']}%")
            with col4:
                st.metric("Revenue", f"${data['total_revenue']:,.2f}")
            
            st.success("Dashboard loaded successfully")
        else:
            st.error("Failed to load dashboard")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Create Order
elif menu == "Create Order":
    st.title("📝 Create New Order")
    
    with st.form("order_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, step=1)
            pickup_address = st.text_area("Pickup Address", height=100)
            package_weight = st.number_input("Package Weight (kg)", min_value=0.1, step=0.1)
        
        with col2:
            delivery_address = st.text_area("Delivery Address", height=100)
            cod_amount = st.number_input("COD Amount ($)", min_value=0.0, step=0.01)
            package_dimensions = st.text_input("Dimensions (optional)")
        
        submitted = st.form_submit_button("Create Order")
        
        if submitted:
            order_data = {
                "customer_id": customer_id,
                "pickup_address": pickup_address,
                "delivery_address": delivery_address,
                "package_weight": package_weight,
                "cod_amount": cod_amount,
                "package_dimensions": package_dimensions
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/orders",
                    json=order_data,
                    headers=get_headers()
                )
                
                if response.status_code == 201:
                    result = response.json()
                    st.success(f"✅ Order created successfully!")
                    st.info(f"AWB Number: {result['awb_number']}")
                    st.info(f"Total Amount: ${result['total_amount']}")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Track Shipment
elif menu == "Track Shipment":
    st.title("🔍 Track Your Shipment")
    
    awb_number = st.text_input("Enter AWB Number", placeholder="AWB20241115...")
    
    if st.button("Track"):
        try:
            response = requests.get(f"{API_BASE_URL}/tracking/{awb_number}")
            
            if response.status_code == 200:
                data = response.json()
                
                st.success(f"Status: **{data['status'].upper()}**")
                if data['current_location']:
                    st.info(f"📍 Current Location: {data['current_location']}")
                
                st.subheader("Tracking History")
                for event in data['events']:
                    with st.expander(f"{event['type']} - {event['timestamp']}"):
                        st.write(f"**Description:** {event['description']}")
                        if event['location']:
                            st.write(f"**Location:** {event['location']}")
            else:
                st.error("Order not found")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Orders List
elif menu == "Orders List":
    st.title("📋 All Orders")
    st.info("This would display a paginated list of all orders with filters")

# Reports
elif menu == "Reports":
    st.title("📈 Reports & Analytics")
    st.info("Advanced reporting features would be implemented here")


# ============================================================================
# 8. INITIALIZE DATABASE
# ============================================================================

def init_database():
    """Initialize database with tables and indexes"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_database()
    print("Starting FastAPI server...")
    print("Run with: uvicorn main:app --reload")