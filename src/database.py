"""
GUI-Based Fine Management System — data layer
===============================================
Mirrors Section 3.5 / 3.6 of the project report. Backs the vehicle
database with SQLite (instead of MATLAB's struct2table) so records
persist across runs. Seeds itself from data/vehicle_db.json on first run.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "vehicle_fines.db"
SEED_PATH = BASE_DIR / "data" / "vehicle_db.json"


@dataclass
class VehicleRecord:
    license_plate: str
    owner_name: str
    phone_number: str
    occupation: str
    rc_number: str
    license_validity: str
    area: str
    violations: str  # comma-separated
    total_fines: int
    status: str


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False) -> None:
    """Create the vehicles table and seed it from vehicle_db.json if empty."""
    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            license_plate     TEXT PRIMARY KEY,
            owner_name        TEXT,
            phone_number      TEXT,
            occupation        TEXT,
            rc_number         TEXT,
            license_validity  TEXT,
            area              TEXT,
            violations        TEXT,
            total_fines       INTEGER DEFAULT 0,
            status            TEXT DEFAULT 'Paid'
        )
        """
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    if count == 0 and SEED_PATH.exists():
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            seed = json.load(f)
        for row in seed:
            conn.execute(
                """
                INSERT OR IGNORE INTO vehicles
                (license_plate, owner_name, phone_number, occupation, rc_number,
                 license_validity, area, violations, total_fines, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["license_plate"],
                    row["owner_name"],
                    row["phone_number"],
                    row["occupation"],
                    row["rc_number"],
                    row["license_validity"],
                    row["area"],
                    ",".join(row.get("violations", [])),
                    row.get("total_fines", 0),
                    row.get("status", "Paid"),
                ),
            )
        conn.commit()
    conn.close()


def get_vehicle(license_plate: str) -> Optional[VehicleRecord]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM vehicles WHERE license_plate = ?", (license_plate,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return VehicleRecord(**dict(row))


def upsert_vehicle(record: VehicleRecord) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO vehicles
        (license_plate, owner_name, phone_number, occupation, rc_number,
         license_validity, area, violations, total_fines, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(license_plate) DO UPDATE SET
            owner_name=excluded.owner_name,
            phone_number=excluded.phone_number,
            occupation=excluded.occupation,
            rc_number=excluded.rc_number,
            license_validity=excluded.license_validity,
            area=excluded.area,
            violations=excluded.violations,
            total_fines=excluded.total_fines,
            status=excluded.status
        """,
        (
            record.license_plate,
            record.owner_name,
            record.phone_number,
            record.occupation,
            record.rc_number,
            record.license_validity,
            record.area,
            record.violations,
            record.total_fines,
            record.status,
        ),
    )
    conn.commit()
    conn.close()


def issue_fine(license_plate: str, violation: str, amount: int, owner_details: dict) -> VehicleRecord:
    """
    Add a violation + fine amount to a vehicle. Creates a new record with
    the supplied owner_details if the plate isn't already in the database
    (mirrors the MATLAB GUI's addFine callback).
    """
    existing = get_vehicle(license_plate)
    if existing is None:
        record = VehicleRecord(
            license_plate=license_plate,
            owner_name=owner_details.get("owner_name", ""),
            phone_number=owner_details.get("phone_number", ""),
            occupation=owner_details.get("occupation", ""),
            rc_number=owner_details.get("rc_number", "New"),
            license_validity=owner_details.get("license_validity", "N/A"),
            area=owner_details.get("area", ""),
            violations=violation,
            total_fines=amount,
            status="Unpaid",
        )
    else:
        violations = [v for v in existing.violations.split(",") if v]
        violations.append(violation)
        record = VehicleRecord(
            license_plate=existing.license_plate,
            owner_name=existing.owner_name,
            phone_number=existing.phone_number,
            occupation=existing.occupation,
            rc_number=existing.rc_number,
            license_validity=existing.license_validity,
            area=existing.area,
            violations=",".join(violations),
            total_fines=existing.total_fines + amount,
            status="Unpaid",
        )
    upsert_vehicle(record)
    return record


def pay_fine(license_plate: str) -> Optional[VehicleRecord]:
    existing = get_vehicle(license_plate)
    if existing is None:
        return None
    existing.total_fines = 0
    existing.status = "Paid"
    upsert_vehicle(existing)
    return existing


def get_all_vehicles() -> List[VehicleRecord]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM vehicles").fetchall()
    conn.close()
    return [VehicleRecord(**dict(r)) for r in rows]
