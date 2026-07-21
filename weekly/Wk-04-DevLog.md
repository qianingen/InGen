The schema separates device, session, sensor_reading, and alert because each table represents a different entity. Device-level metadata is stored once in the device table instead of being repeated across every telemetry row. Session-level metadata, such as source file and row count, is stored in the session table. Timestamp-level telemetry facts and derived features are stored in sensor_reading. Alert events are stored separately because multiple alerts can be associated with a device, session, or reading.

This avoids update anomalies. For example, changing a device name only requires updating one row in the device table rather than thousands of sensor_reading rows. The schema also avoids insert and delete anomalies because devices and sessions can exist independently of individual readings.

The schema follows 3NF because non-key fields in each table depend on the key, the whole key, and nothing but the key. Device metadata depends on device_id. Session metadata depends on session_id. Sensor values and derived features depend on reading_id. Alert metadata depends on alert_id.

erDiagram
    DEVICE ||--o{ SESSION : has
    DEVICE ||--o{ SENSOR_READING : produces
    SESSION ||--o{ SENSOR_READING : contains
    SENSOR_READING ||--o{ ALERT : triggers
    SESSION ||--o{ ALERT : contains
    DEVICE ||--o{ ALERT : raises

    DEVICE {
        string device_id PK
        string device_name
        string product_anchor
        bigint created_at_ms
    }

    SESSION {
        string session_id PK
        string device_id FK
        string source_file
        bigint started_at_ms
        bigint ended_at_ms
        int row_count
        bigint created_at_ms
    }

    SENSOR_READING {
        string reading_id PK
        string device_id FK
        string session_id FK
        bigint timestamp_ms
        float lat
        float lon
        float battery_soc
        float lidar_distance_m
        float wheel_imbalance_score
        float composite_health_score
    }

    ALERT {
        string alert_id PK
        string device_id FK
        string session_id FK
        string reading_id FK
        string alert_type
        int severity
        bigint detected_at_ms
    }