package com.mcbuild.mod;

import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Parses mcbuild's mc_id strings (e.g. {@code "minecraft:oak_stairs[facing=north,half=top]"})
 * into a {@link BlockState} by reusing vanilla's own command-argument parser, so the grammar
 * mcbuild emits (see {@code src/mcbuild/palette.py::_mc_id}) can never drift from what
 * Minecraft itself accepts.
 */
public final class BlockStateCodec {
    private BlockStateCodec() {}

    public static BlockState parse(HolderLookup.Provider registries, String mcId) throws CommandSyntaxException {
        HolderLookup<Block> blockLookup = registries.lookupOrThrow(Registries.BLOCK);
        BlockStateParser.BlockResult result = BlockStateParser.parseForBlock(blockLookup, mcId, false);
        return result.blockState();
    }
}
