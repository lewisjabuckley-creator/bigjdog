"""Relational schema for JARVIS's structured state (spec §100).

Structured state lives in structured storage. Frequently queried attributes are
real columns; evolving detail lives in a JSON ``data``/``attrs`` column so the
schema can grow without constant migrations. Text retrieval uses FTS5; vectors
are an optional side table, never the primary store.
"""

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            severity INTEGER NOT NULL,
            entity_id TEXT,
            task_id TEXT,
            payload TEXT NOT NULL
        );
        CREATE INDEX idx_events_ts ON events(ts);
        CREATE INDEX idx_events_type ON events(type, ts);
        CREATE INDEX idx_events_task ON events(task_id);

        CREATE TABLE state (
            key TEXT PRIMARY KEY,
            fact TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            attrs TEXT NOT NULL DEFAULT '{}',
            source TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX idx_entities_type ON entities(type);
        CREATE INDEX idx_entities_name ON entities(name COLLATE NOCASE);

        CREATE TABLE relations (
            src TEXT NOT NULL,
            rel TEXT NOT NULL,
            dst TEXT NOT NULL,
            attrs TEXT NOT NULL DEFAULT '{}',
            updated_at REAL NOT NULL,
            PRIMARY KEY (src, rel, dst)
        );
        CREATE INDEX idx_relations_dst ON relations(dst, rel);

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            objective TEXT NOT NULL,
            owner TEXT NOT NULL,
            created_by TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            status_reason TEXT,
            outcome TEXT,
            progress REAL NOT NULL DEFAULT 0,
            project_id TEXT,
            parent_id TEXT,
            data TEXT NOT NULL,
            deadline REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            version INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_tasks_status ON tasks(status, priority, created_at);

        CREATE TABLE approvals (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            step_id TEXT,
            tool TEXT NOT NULL,
            args TEXT NOT NULL,
            summary TEXT NOT NULL,
            risk INTEGER NOT NULL,
            level INTEGER NOT NULL,
            reason TEXT,
            requested_by TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            decided_at REAL,
            decided_by TEXT,
            expires_at REAL
        );
        CREATE INDEX idx_approvals_status ON approvals(status, created_at);

        CREATE TABLE grants (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            level INTEGER NOT NULL,
            tools TEXT NOT NULL,
            paths TEXT NOT NULL,
            project_id TEXT,
            task_id TEXT,
            expires_at REAL,
            max_uses INTEGER,
            uses INTEGER NOT NULL DEFAULT 0,
            reason TEXT,
            created_by TEXT NOT NULL,
            created_at REAL NOT NULL,
            revoked_at REAL
        );

        CREATE TABLE audit (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            actor TEXT NOT NULL,
            task_id TEXT,
            action TEXT NOT NULL,
            tool TEXT,
            params TEXT,
            ok INTEGER,
            outcome TEXT,
            summary TEXT,
            verification TEXT,
            authorization TEXT,
            reason TEXT,
            model TEXT,
            agent TEXT,
            rollback TEXT,
            duration_s REAL,
            dry_run INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_audit_ts ON audit(ts);
        CREATE INDEX idx_audit_task ON audit(task_id);

        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            subject TEXT,
            project_id TEXT,
            user_id TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            provenance TEXT,
            confidence TEXT,
            importance REAL NOT NULL DEFAULT 0.5,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_accessed REAL,
            expires_at REAL
        );
        CREATE INDEX idx_memories_kind ON memories(kind, project_id);
        CREATE VIRTUAL TABLE memories_fts USING fts5(memory_id UNINDEXED, content, subject, tags);

        CREATE TABLE embeddings (
            memory_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL
        );

        CREATE TABLE decisions (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            project_id TEXT,
            title TEXT NOT NULL,
            decision TEXT NOT NULL,
            context TEXT,
            alternatives TEXT NOT NULL DEFAULT '[]',
            reason TEXT,
            user_involvement TEXT,
            outcome TEXT,
            tags TEXT NOT NULL DEFAULT '[]'
        );
        CREATE VIRTUAL TABLE decisions_fts USING fts5(decision_id UNINDEXED, title, decision, context, reason);

        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            root TEXT,
            description TEXT,
            policy TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_opened REAL
        );

        CREATE TABLE notifications (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            priority INTEGER NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            source TEXT,
            task_id TEXT,
            dedupe_key TEXT,
            count INTEGER NOT NULL DEFAULT 1,
            state TEXT NOT NULL,
            expires_at REAL,
            delivered_at REAL,
            acknowledged_at REAL
        );
        CREATE INDEX idx_notifications_state ON notifications(state, priority);

        CREATE TABLE automations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            spec TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            owner TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_run REAL,
            next_run REAL,
            run_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE conversation (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_conversation_ts ON conversation(ts);

        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            prefs TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        """,
    ),
    (
        2,
        """
        ALTER TABLE tasks ADD COLUMN idempotency_key TEXT;
        CREATE UNIQUE INDEX idx_tasks_idempotency ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;

        ALTER TABLE automations ADD COLUMN last_status TEXT;
        ALTER TABLE automations ADD COLUMN last_error TEXT;
        ALTER TABLE automations ADD COLUMN last_task_id TEXT;
        ALTER TABLE automations ADD COLUMN missed INTEGER NOT NULL DEFAULT 0;

        CREATE TABLE runtime_runs (
            id TEXT PRIMARY KEY,
            pid INTEGER NOT NULL,
            mode TEXT NOT NULL,
            version TEXT,
            host TEXT,
            started_at REAL NOT NULL,
            heartbeat_at REAL,
            stopped_at REAL,
            clean INTEGER NOT NULL DEFAULT 0,
            info TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_runtime_runs_started ON runtime_runs(started_at);

        CREATE TABLE requests (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            response TEXT,
            created_at REAL NOT NULL,
            finished_at REAL
        );
        CREATE INDEX idx_requests_session ON requests(session_id, created_at);

        CREATE TABLE briefings (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            data TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX idx_briefings_ts ON briefings(ts);
        CREATE INDEX idx_conversation_session ON conversation(session_id, ts);
        """,
    ),
]
