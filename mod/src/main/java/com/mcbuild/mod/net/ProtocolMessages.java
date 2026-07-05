package com.mcbuild.mod.net;

import java.util.List;

/**
 * Gson-mapped mirrors of mcbuild's {@code src/mcbuild/server/protocol.py} message shapes.
 * Keep these in sync when the Python side's wire contract changes.
 */
public final class ProtocolMessages {
    private ProtocolMessages() {}

    /** Every message has at least a "type" field; check it before picking the concrete shape. */
    public static class Envelope {
        public String type;
    }

    public static class Hello extends Envelope {
        public String run_dir;
    }

    public static class Chat extends Envelope {
        public String text;
    }

    public static class Blocks extends Envelope {
        public int iteration;
        public List<BlockChange> changes;
    }

    public static class BlockChange {
        public int x;
        public int y;
        public int z;
        /** Full "minecraft:base[state]" id, or null for a removed cell (place air). */
        public String block;
    }

    public static class Finish extends Envelope {
        public String summary;
    }

    public static class Error extends Envelope {
        public String reason;
    }
}
