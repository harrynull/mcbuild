package com.mcbuild.mod;

import com.mcbuild.mod.net.ProtocolMessages;
import com.mcbuild.mod.net.WsClient;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.neoforged.neoforge.server.ServerLifecycleHooks;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Tracks the single in-flight {@code /build} session (v1 supports one at a time) and applies
 * streamed block deltas + chat feedback on behalf of the connected {@link WsClient}.
 */
public final class BuildSessionManager {
    private static Session active;

    private BuildSessionManager() {}

    public static synchronized boolean startBuild(String prompt, BlockPos anchor, ServerLevel level, ServerPlayer player) {
        if (active != null) {
            return false;
        }
        Session session = new Session(anchor, level, player);
        active = session;
        WsClient client = new WsClient(prompt, session);
        session.client = client;
        client.connect();
        return true;
    }

    private static synchronized void clear(Session session) {
        if (active == session) {
            active = null;
        }
    }

    /** Per-build state: anchor position, target level/player, and the owning WS connection. */
    public static final class Session {
        private final BlockPos anchor;
        private final ServerLevel level;
        private final ServerPlayer player;
        private WsClient client;

        private Session(BlockPos anchor, ServerLevel level, ServerPlayer player) {
            this.anchor = anchor;
            this.level = level;
            this.player = player;
        }

        public void onChat(String text) {
            runOnServerThread(() -> player.sendSystemMessage(Component.literal(text)));
        }

        public void onBlocks(ProtocolMessages.Blocks blocks) {
            runOnServerThread(() -> {
                for (ProtocolMessages.BlockChange change : blocks.changes) {
                    BlockPos pos = anchor.offset(change.x, change.y, change.z);
                    BlockState state;
                    if (change.block == null) {
                        state = Blocks.AIR.defaultBlockState();
                    } else {
                        try {
                            state = BlockStateCodec.parse(level.registryAccess(), change.block);
                        } catch (CommandSyntaxException e) {
                            player.sendSystemMessage(Component.literal(
                                    "[mcbuild] failed to parse block '" + change.block + "': " + e.getMessage()));
                            continue;
                        }
                    }
                    level.setBlock(pos, state, 3);
                }
            });
        }

        public void onFinish(String summary) {
            runOnServerThread(() -> player.sendSystemMessage(Component.literal("[mcbuild] " + summary)));
            clear(this);
        }

        public void onError(String reason) {
            runOnServerThread(() -> player.sendSystemMessage(Component.literal("[mcbuild] error: " + reason)));
            clear(this);
        }

        public void onClose() {
            clear(this);
        }

        private void runOnServerThread(Runnable task) {
            MinecraftServer server = ServerLifecycleHooks.getCurrentServer();
            if (server != null) {
                server.execute(task);
            }
        }
    }
}
