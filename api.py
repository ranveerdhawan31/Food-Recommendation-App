"""Thin FastAPI transport over the pure-Python core. NOT exercised by the unit
tests (fastapi was unavailable when this was written). Install fastapi + uvicorn
and run:  uvicorn foodrec.api:app

Group sessions live in memory here; in production keep session state in Redis
(serialize `GroupSession` inputs, rebuild on reconnect) so any ECS task can serve
a WebSocket.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .group_fusion import GroupSession

app = FastAPI(title="foodrec")
SESSIONS: dict[str, GroupSession] = {}  # session id -> session (replace with Redis)
CONNECTIONS: dict[str, set[WebSocket]] = {}


@app.websocket("/ws/group/{session_id}")
async def group_ws(ws: WebSocket, session_id: str) -> None:
    session = SESSIONS.get(session_id)
    if session is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    CONNECTIONS.setdefault(session_id, set()).add(ws)
    try:
        while True:
            event = await ws.receive_json()
            reply = session.handle_event(event)  # validates member and safe-pool membership
            for peer in list(CONNECTIONS[session_id]):
                await peer.send_json(reply)  # broadcast so every member sees the same options/votes
    except WebSocketDisconnect:
        CONNECTIONS[session_id].discard(ws)
