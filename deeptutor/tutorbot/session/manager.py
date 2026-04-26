"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from deeptutor.tutorbot.config.paths import get_legacy_sessions_dir
from deeptutor.tutorbot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a user turn."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.tutorbot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        # Auto-update index for bot web sessions
        parts = session.key.split(":")
        if len(parts) == 4 and parts[0] == "bot" and parts[2] == "web":
            # key format: bot:{bot_id}:web:{session_id}
            bot_id = parts[1]
            sid = parts[3]
            index = self._load_index()
            if session.key in index:
                index[session.key]["updated_at"] = session.updated_at.isoformat()
                index[session.key]["message_count"] = len(session.messages)
                self._save_index(index)

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    # --- Bot multi-session index (sessions_index.json) ---

    def _get_index_path(self) -> Path:
        """Get path to sessions index file."""
        return self.sessions_dir / "sessions_index.json"

    def _load_index(self) -> dict[str, Any]:
        """Load session metadata index from disk."""
        path = self._get_index_path()
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: dict[str, Any]) -> None:
        """Save session metadata index to disk."""
        path = self._get_index_path()
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    # --- Bot session CRUD ---

    def list_bot_sessions(self, bot_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """List all web sessions for a specific bot, sorted by updated_at desc.

        Includes legacy (pre-multi-session) JSONL files as a 'default' entry
        so old chat history is not lost.
        """
        prefix = f"bot:{bot_id}:web:"
        legacy_key = f"bot:{bot_id}"
        legacy_session_id = "default"
        index = self._load_index()
        result = []

        # --- New-format sessions from index ---
        for key, meta in index.items():
            if key.startswith(prefix):
                result.append({
                    "session_id": key[len(prefix):],
                    "title": meta.get("title", "New Chat"),
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                    "message_count": meta.get("message_count", 0),
                })

        # --- Legacy session (old format bot:{bot_id}.jsonl) ---
        legacy_path = self._get_session_path(legacy_key)
        if legacy_path.exists() and legacy_key not in index:
            # Count messages (skip metadata line)
            msg_count = 0
            try:
                with open(legacy_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("_type") != "metadata":
                            msg_count += 1
            except Exception:
                pass

            # Get timestamps from metadata line
            created_at = ""
            updated_at = ""
            try:
                with open(legacy_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            created_at = data.get("created_at", "")
                            updated_at = data.get("updated_at", "")
            except Exception:
                pass

            result.append({
                "session_id": legacy_session_id,
                "title": "Default Chat",
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": msg_count,
                "_is_legacy": True,
            })

        result.sort(key=lambda x: x["updated_at"], reverse=True)
        return result[:limit]

    def create_bot_session(self, bot_id: str, session_id: str, title: str | None = None) -> Session:
        """Create a new empty session for a bot's web chat."""
        key = f"bot:{bot_id}:web:{session_id}"
        session = Session(key=key)
        now = datetime.now().isoformat()

        # Write empty JSONL so it appears on disk
        self.save(session)

        # Update index
        index = self._load_index()
        index[key] = {
            "title": title or "New Chat",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
        self._save_index(index)

        return session

    def rename_bot_session(self, bot_id: str, session_id: str, title: str) -> bool:
        """Rename a bot session (updates index only)."""
        key = f"bot:{bot_id}:web:{session_id}"
        index = self._load_index()
        if key not in index:
            return False
        index[key]["title"] = title
        index[key]["updated_at"] = datetime.now().isoformat()
        self._save_index(index)
        return True

    def delete_bot_session(self, bot_id: str, session_id: str) -> bool:
        """Delete a bot session: remove from cache, delete JSONL, remove from index."""
        key = f"bot:{bot_id}:web:{session_id}"

        # Remove from cache
        self.invalidate(key)

        # Delete JSONL file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()

        # Remove from index
        index = self._load_index()
        if key in index:
            del index[key]
            self._save_index(index)
            return True
        return False

    def update_session_stats(self, bot_id: str, session_id: str, delta: int = 1) -> None:
        """Update message count and updated_at timestamp in index."""
        key = f"bot:{bot_id}:web:{session_id}"
        index = self._load_index()
        if key in index:
            index[key]["message_count"] = index[key].get("message_count", 0) + delta
            index[key]["updated_at"] = datetime.now().isoformat()
            self._save_index(index)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
