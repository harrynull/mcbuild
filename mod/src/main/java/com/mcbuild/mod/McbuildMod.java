package com.mcbuild.mod;

import com.mcbuild.mod.command.BuildCommand;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.common.EventBusSubscriber;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.event.RegisterCommandsEvent;

/** Mod entrypoint: wires the /build command into the server command dispatcher. */
@Mod(McbuildMod.MOD_ID)
public class McbuildMod {
    public static final String MOD_ID = "mcbuild_live";

    @EventBusSubscriber(modid = MOD_ID)
    public static class GameEvents {
        @SubscribeEvent
        public static void onRegisterCommands(RegisterCommandsEvent event) {
            BuildCommand.register(event.getDispatcher());
        }
    }
}
