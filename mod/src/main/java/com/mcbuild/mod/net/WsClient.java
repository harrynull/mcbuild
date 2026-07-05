package com.mcbuild.mod.net;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mcbuild.mod.BuildSessionManager;
import java.net.URI;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;

/**
 * One connection per {@code /build} invocation: sends the prompt, then dispatches streamed
 * protocol messages to the owning {@link BuildSessionManager.Session}.
 *
 * <p>All {@code on*} callbacks here fire on this client's own network thread, not the server
 * thread -- {@link BuildSessionManager.Session} is responsible for hopping back onto the
 * server thread before touching any {@code Level}/{@code Player} state.
 */
public class WsClient extends WebSocketClient {
    private static final Gson GSON = new Gson();
    private static final URI SERVER_URI = URI.create("ws://127.0.0.1:8765");

    private final String prompt;
    private final BuildSessionManager.Session session;

    public WsClient(String prompt, BuildSessionManager.Session session) {
        super(SERVER_URI);
        this.prompt = prompt;
        this.session = session;
    }

    @Override
    public void onOpen(ServerHandshake handshake) {
        JsonObject request = new JsonObject();
        request.addProperty("cmd", "build");
        request.addProperty("prompt", prompt);
        send(request.toString());
    }

    @Override
    public void onMessage(String message) {
        JsonObject envelope = GSON.fromJson(message, JsonObject.class);
        if (envelope == null || !envelope.has("type")) {
            return;
        }
        String type = envelope.get("type").getAsString();
        switch (type) {
            case "hello" -> {
                // informational only; no action needed.
            }
            case "chat" -> session.onChat(GSON.fromJson(message, ProtocolMessages.Chat.class).text);
            case "blocks" -> session.onBlocks(GSON.fromJson(message, ProtocolMessages.Blocks.class));
            case "finish" -> {
                session.onFinish(GSON.fromJson(message, ProtocolMessages.Finish.class).summary);
                close();
            }
            case "error" -> {
                session.onError(GSON.fromJson(message, ProtocolMessages.Error.class).reason);
                close();
            }
            default -> {
                // unknown message type; ignore rather than crash the session.
            }
        }
    }

    @Override
    public void onClose(int code, String reason, boolean remote) {
        session.onClose();
    }

    @Override
    public void onError(Exception ex) {
        session.onError("WebSocket error: " + ex.getMessage());
    }
}
