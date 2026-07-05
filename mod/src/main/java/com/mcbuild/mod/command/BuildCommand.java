package com.mcbuild.mod.command;

import com.mcbuild.mod.BuildSessionManager;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;

/**
 * {@code /build <prompt>}: anchors a live mcbuild session at the invoking player's position
 * and starts streaming block placements from the Python agent. No permission is required
 * (auth is intentionally out of scope for this feature); only one build may run at a time,
 * enforced by {@link BuildSessionManager}.
 */
public final class BuildCommand {
    private BuildCommand() {}

    public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(
                Commands.literal("build")
                        .requires(source -> source.hasPermission(0))
                        .then(Commands.argument("prompt", StringArgumentType.greedyString())
                                .executes(BuildCommand::run)));
    }

    private static int run(com.mojang.brigadier.context.CommandContext<CommandSourceStack> ctx) {
        CommandSourceStack source = ctx.getSource();
        ServerPlayer player;
        try {
            player = source.getPlayerOrException();
        } catch (com.mojang.brigadier.exceptions.CommandSyntaxException e) {
            source.sendFailure(Component.literal("[mcbuild] /build must be run by a player."));
            return 0;
        }
        ServerLevel level = source.getLevel();
        BlockPos anchor = player.blockPosition();
        String prompt = StringArgumentType.getString(ctx, "prompt");

        if (!BuildSessionManager.startBuild(prompt, anchor, level, player)) {
            source.sendFailure(Component.literal("[mcbuild] A build is already in progress."));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal("[mcbuild] Building \"" + prompt + "\" at " + anchor.toShortString() + "..."),
                false);
        return 1;
    }
}
