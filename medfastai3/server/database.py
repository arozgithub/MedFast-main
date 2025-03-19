from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import sys

# Load database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:edifire11@localhost:5432/medfast")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# Create the database engine with connection pooling
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    connection = engine.connect()
    connection.close()
    print("✅ Database connection successful.")
except Exception as e:
    print(f"❌ Database connection failed: {e}", file=sys.stderr)
    sys.exit(1)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def init_db():
    """Initialize database and create tables"""
    print("🔄 Initializing database...")
    try:
        from server.models import User  # Ensure models are imported before creating tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialized successfully.")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

def get_db():
    """Dependency to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
