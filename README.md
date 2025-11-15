# 📦 Courier Service ERP System - Complete Documentation

## 🎯 Project Overview

A **production-ready Enterprise Resource Planning (ERP)** system specifically designed for courier and logistics companies. This comprehensive solution integrates order management, real-time tracking, fleet operations, financial management, and human resources into a unified platform.

---

## 🏗️ System Architecture

### **Technology Stack**

#### **Backend (Microservices)**
- **FastAPI** - High-performance REST API framework
- **SQLAlchemy** - ORM for database operations with advanced query optimization
- **PostgreSQL** - Relational database with indexing strategies
- **Pydantic** - Data validation and serialization
- **JWT Authentication** - Secure token-based authentication
- **Uvicorn** - ASGI server for production deployment

#### **Frontend**
- **Streamlit** - Interactive web application interface
- **Responsive UI** - Multi-page navigation with role-based access

#### **Security & Performance**
- **Bcrypt** - Password hashing
- **OAuth2** - Industry-standard authentication
- **Connection Pooling** - Optimized database connections (20 base + 40 overflow)
- **Database Indexing** - Strategic indexes on high-query columns
- **Input Validation** - Comprehensive data validation at all entry points

---

## 🚀 Core Features

### **1. Order Management System**
- **Automated AWB Generation** - Unique tracking numbers with timestamp + random suffix
- **Smart Pricing Engine** - Weight and distance-based calculation
- **Bulk Order Processing** - Handle high-volume operations
- **Address Validation** - Minimum length and format checks
- **COD Integration** - Cash on Delivery tracking and reconciliation
- **Multi-status Workflow** - 7 status states (Pending → Delivered/Failed/Returned)

### **2. Real-Time Tracking & Visibility**
- **GPS Tracking Events** - Location-based updates with coordinates
- **Public Tracking Portal** - No-login AWB lookup for customers
- **Event Timeline** - Complete shipment history with timestamps
- **Status Notifications** - Automated updates at each checkpoint
- **Current Location Display** - Live package positioning

### **3. Delivery & Dispatch Management**
- **Driver Assignment** - Link orders to delivery personnel
- **Vehicle Management** - Track fleet capacity, mileage, maintenance
- **Route Optimization** - Distance and sequence calculation
- **Digital Proof of Delivery (POD)**:
  - E-signature capture
  - Photo documentation
  - GPS geo-stamping
  - Delivery notes

### **4. Financial Integration**
- **Automated Invoicing** - Generate invoices upon delivery completion
- **Tax Calculation** - Built-in 18% tax computation
- **Payment Tracking** - Monitor paid/pending status
- **Revenue Reporting** - Real-time financial analytics
- **COD Reconciliation** - Track collected cash vs remittances

### **5. User & Role Management**
- **Role-Based Access Control (RBAC)**:
  - **Admin** - Full system access
  - **Manager** - Operations oversight
  - **Driver** - Delivery execution
  - **Customer** - Order creation & tracking
- **JWT Token Authentication** - Secure API access
- **Session Management** - Automatic timeout handling
- **User Profiles** - Personal information and preferences

### **6. Dashboard & Analytics**
- **KPI Metrics**:
  - Total Orders
  - Delivery Rate
  - Revenue Tracking
  - Pending Orders Count
- **Performance Indicators** - On-time delivery percentages
- **Real-time Updates** - Live data refresh

### **7. Employee & HR Integration**
- **Employee Records** - Link users to HR data
- **Performance Tracking** - Delivery success rates
- **Salary Management** - Base pay + bonus calculations
- **Compliance Tracking** - License expiry, certifications
- **Department Organization** - Structure by role/location

---

## 🗄️ Database Schema

### **Optimized Data Model**

#### **Core Tables**
1. **users** - Authentication and user profiles
2. **customers** - Customer information and contacts
3. **orders** - Shipment orders with full lifecycle
4. **deliveries** - Driver assignments and completion data
5. **vehicles** - Fleet management and maintenance
6. **tracking_events** - Event-based tracking history
7. **invoices** - Financial billing records
8. **employees** - HR and workforce data

#### **Performance Features**
- **Strategic Indexing**:
  - Composite indexes on status + date columns
  - Foreign key indexes for join optimization
  - Unique constraints on AWB/invoice numbers
- **Check Constraints** - Data integrity (positive weights, amounts)
- **Cascade Deletes** - Automatic cleanup of related records
- **Connection Pooling** - Pre-ping health checks

---

## 🔐 Security Implementation

### **Authentication & Authorization**
- **JWT Tokens** - 30-minute expiration with refresh capability
- **Password Hashing** - Bcrypt with salt
- **Role Validation** - Server-side permission checks
- **CORS Protection** - Configurable origin policies

### **Input Validation**
- **Email Format** - Regex pattern validation
- **Address Length** - Minimum 10 characters
- **Weight Constraints** - 0.1kg to 1000kg range
- **Amount Checks** - Non-negative financial values
- **SQL Injection Prevention** - ORM parameterized queries

### **API Security**
- **Bearer Token Headers** - Required for protected endpoints
- **401 Unauthorized** - Invalid/expired tokens
- **403 Forbidden** - Inactive accounts
- **Rate Limiting Ready** - Structure for request throttling

---

## 📊 API Endpoints

### **Authentication**
```
POST   /api/auth/login      - Login with credentials
POST   /api/auth/register   - Register new user
GET    /api/auth/me         - Get current user info
```

### **Orders**
```
POST   /api/orders                - Create new order
GET    /api/orders/{id}           - Get order details
PUT    /api/orders/{id}/status    - Update order status
POST   /api/orders/{id}/invoice   - Generate invoice
```

### **Tracking**
```
GET    /api/tracking/{awb}        - Public tracking (no auth)
```

### **Reports**
```
GET    /api/reports/dashboard     - Dashboard statistics
```

---

## 🎭 Demo User Credentials

The system includes **4 pre-configured test users** for immediate testing:

| Role | Email | Password | Use Case |
|------|-------|----------|----------|
| **Admin** | admin@courier.com | admin123 | System configuration, full access |
| **Manager** | manager@courier.com | manager123 | Operations monitoring, reporting |
| **Driver** | driver@courier.com | driver123 | Delivery updates, POD capture |
| **Customer** | customer@courier.com | customer123 | Order placement, tracking |

---

## 🎨 User Interface Features

### **Login Experience**
- Clean authentication form
- Demo credentials display
- Quick login buttons for each role
- Real-time validation feedback

### **Dashboard**
- Welcome message with user name
- 4-metric KPI cards with trend indicators
- Color-coded status badges
- Responsive grid layout

### **Order Creation**
- Two-column form layout
- Real-time weight/cost calculation
- Validation error messages
- Success confirmation with AWB display

### **Tracking Interface**
- Large search input
- Status color coding (🟡🔵🟠🟣🟢🔴)
- Expandable event timeline
- Location breadcrumbs

### **Navigation**
- Sidebar menu with icons
- User profile display
- Role badge
- One-click logout

---

## 🔧 Installation & Setup

### **Prerequisites**
```bash
Python 3.8+
PostgreSQL 12+
pip package manager
```

### **Step 1: Install Dependencies**
```bash
pip install fastapi uvicorn[standard] streamlit sqlalchemy asyncpg \
            pydantic email-validator python-jose[cryptography] \
            passlib[bcrypt] python-multipart requests pandas
```

### **Step 2: Database Setup**
```bash
# Update DATABASE_URL in the code
DATABASE_URL = "postgresql://username:password@localhost/courier_erp"

# Initialize database
python courier.py
```

### **Step 3: Start Backend**
```bash
uvicorn courier:app --reload --port 8000
```

### **Step 4: Start Frontend**
```bash
streamlit run streamlit_app.py
```

### **Step 5: Access Application**
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📈 Business Logic Services (OOP Design)

### **OrderService**
- AWB generation with uniqueness guarantee
- Dynamic pricing calculation
- Order lifecycle management
- Status update orchestration

### **TrackingService**
- Event creation and logging
- Timeline reconstruction
- Public tracking queries

### **InvoiceService**
- Invoice number generation
- Tax computation (18% configurable)
- Payment status tracking
- Duplicate prevention

### **RouteOptimizationService**
- Driver assignment logic
- Distance calculation
- Sequence optimization (extensible)

---

## 🎯 Production Readiness Features

### **Scalability**
- Microservices architecture - Easy horizontal scaling
- Stateless API design - Load balancer compatible
- Database connection pooling - Handle 60 concurrent connections
- Async operations ready - Non-blocking I/O support

### **Monitoring & Logging**
- Structured error handling
- HTTP status code standards
- Exception tracking points
- Query performance insights (via SQLAlchemy echo)

### **Extensibility**
- Plugin-ready service classes
- Middleware support (CORS configured)
- Environment-based configuration
- API versioning structure

### **Data Integrity**
- Foreign key constraints
- Check constraints on critical fields
- Transaction management via ORM
- Rollback on error

---

## 🚦 Workflow Example

### **Complete Order Lifecycle**

1. **Customer** logs in and creates order
   - System generates AWB: `AWB20241115123456789012`
   - Calculates price: `$5 base + $2/kg + $0.5/km`
   - Status: `PENDING`

2. **Manager** assigns driver
   - Links order to driver via `Delivery` record
   - Status: `PICKED_UP`
   - Creates tracking event

3. **Driver** updates location
   - Mobile app sends GPS coordinates
   - Status: `IN_TRANSIT` → `OUT_FOR_DELIVERY`
   - Multiple tracking events logged

4. **Delivery completion**
   - Driver captures signature + photo
   - GPS coordinates stamped
   - Status: `DELIVERED`
   - Timestamp recorded

5. **System auto-generates invoice**
   - Invoice number: `INV20241115000123`
   - Calculates: Subtotal + Tax (18%) + COD
   - Payment status: `PENDING`

6. **Financial reconciliation**
   - Invoice marked `PAID`
   - Revenue added to reports
   - COD amount tracked separately

---

## 🎓 Key Technical Highlights

### **OOP Principles**
- Service-oriented architecture
- Single Responsibility Principle (each service handles one domain)
- Dependency Injection via FastAPI
- Clear separation of concerns (DB models, schemas, services, routes)

### **Database Optimization**
- **Compound Indexes**: `(status, created_at)`, `(driver_id, assigned_at)`
- **Selective Loading**: Only fetch required columns
- **Lazy Loading**: Related objects loaded on-demand
- **Query Optimization**: Aggregate functions instead of Python loops

### **Security Best Practices**
- Never store plain passwords
- Token expiration enforcement
- HTTPS-ready (add SSL in production)
- Input sanitization at model level
- SQL injection prevention via ORM

### **Code Quality**
- Type hints throughout
- Docstrings on all public methods
- Consistent naming conventions
- Error handling with specific exceptions
- Configuration externalization ready

---

## 🔮 Future Enhancements

### **Planned Features**
- [ ] SMS/Email notification service (Twilio/SendGrid integration)
- [ ] Real-time WebSocket updates for tracking
- [ ] Mobile app for drivers (React Native)
- [ ] Advanced route optimization (Google Maps API)
- [ ] Machine learning for delivery time prediction
- [ ] Multi-language support
- [ ] Export reports to PDF/Excel
- [ ] Customer satisfaction ratings
- [ ] Barcode/QR code scanning module
- [ ] Integration with payment gateways

### **Scalability Roadmap**
- Redis caching layer
- Celery background tasks for heavy operations
- Elasticsearch for full-text search
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline (GitHub Actions)

---

## 📝 License & Usage

This system is designed as a **complete reference implementation** for courier/logistics ERP systems. It demonstrates:

- Production-grade Python architecture
- Modern web development patterns
- Database design best practices
- Security implementation standards
- Full-stack integration techniques

**Ideal for:**
- Logistics companies seeking digital transformation
- Developers learning enterprise application design
- Startups building courier platforms
- Educational institutions teaching software engineering

---

## 🤝 Support & Documentation

### **API Documentation**
- Interactive Swagger UI at `/docs`
- ReDoc alternative at `/redoc`
- Auto-generated from Pydantic models

### **Code Comments**
- Inline documentation throughout
- Function docstrings
- Complex logic explanations
- Architecture decision notes

### **Testing Approach**
Use demo credentials to test:
1. Customer flow (order creation + tracking)
2. Driver flow (delivery updates)
3. Manager flow (monitoring operations)
4. Admin flow (system configuration)

---

## 🎉 Summary

This **Courier ERP System** provides a **complete, production-ready foundation** for logistics operations. With its modern architecture, comprehensive features, and security-first design, it serves as both a **functional business application** and a **learning resource** for enterprise software development.

**Key Strengths:**
✅ Complete feature set covering operations, finance, and HR
✅ Optimized database with strategic indexing
✅ Secure authentication and authorization
✅ Clean OOP design with service layer
✅ RESTful API with comprehensive validation
✅ Interactive Streamlit frontend
✅ Ready for production deployment
✅ Extensible microservices architecture

**Start building your logistics empire today! 📦🚀**
