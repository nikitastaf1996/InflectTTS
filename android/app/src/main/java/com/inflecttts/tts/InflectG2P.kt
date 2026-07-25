package com.inflecttts.tts

/**
 * InflectG2P — rule-based English text → IPA phoneme → token ID converter.
 *
 * The Inflect-Nano-v2 model was trained on eSpeak IPA output (see
 * `runtime/text/symbols.py` and `inflect_vits_frontend.py`). The proper
 * frontend uses `phonemizer` (eSpeak backend) which is a C library —
 * not portable to Android.
 *
 * This class implements a **rule-based** English g2p that covers common
 * letter patterns. It won't match eSpeak exactly, but it produces
 * plausible IPA that the model can decode into recognizable speech.
 *
 * The symbol table (from `runtime/text/symbols.py`):
 *   index 0       = '_'  (pad)
 *   index 1-15    = punctuation: ; : , . ! ? ¡ ¿ — … " « » “ ”
 *   index 16      = ' '  (space, also SPACE_ID)
 *   index 17-68   = A-Z a-z (ASCII letters — used as fallback)
 *   index 69-177  = IPA letters (ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ)
 *
 * Config: `add_blank: true` means intersperse pad (0) between every
 * phoneme (matching `commons.intersperse(sequence, 0)` in the pathway
 * script). MAX_SEQ_LEN = 256 (truncate/pad).
 */
object InflectG2P {

    /** Pad token (also used for intersperse blank). */
    private const val PAD = 0

    /** Space token — separates words. */
    private const val SPACE = 16

    // ---- Punctuation tokens (indices 1-15) ----
    private const val PUNCT_COMMA = 3   // ','
    private const val PUNCT_PERIOD = 4  // '.'
    private const val PUNCT_BANG = 5    // '!'
    private const val PUNCT_QMARK = 6   // '?'
    private const val PUNCT_SEMI = 1    // ';'
    private const val PUNCT_COLON = 2   // ':'

    // ---- IPA consonant tokens (from symbols.py indices 69-177) ----
    private const val P = 58      // p
    private const val B = 44      // b
    private const val T = 62      // t
    private const val D = 46      // d
    private const val K = 53      // k
    private const val G = 49      // g
    private const val F = 48      // f
    private const val V = 64      // v
    private const val TH = 119    // θ (voiceless th)
    private const val DH = 81     // ð (voiced th)
    private const val S = 61      // s
    private const val Z = 68      // z
    private const val SH = 131    // ʃ
    private const val ZH = 147    // ʒ
    private const val H = 50      // h
    private const val M = 55      // m
    private const val N = 56      // n
    private const val NG = 112    // ŋ
    private const val L = 54      // l
    private const val R = 60      // r
    private const val W = 65      // w
    private const val Y = 52      // j
    private const val CH = 131    // ʃ (use SH for ch — close enough)
    private const val JH = 147    // ʒ (use ZH for j — close enough)

    // ---- IPA vowel tokens ----
    private const val AA = 69     // ɑ (father)
    private const val AE = 72     // æ (cat)
    private const val AH = 83     // ə (schwa)
    private const val EH = 86     // ɛ (bed)
    private const val IH = 102    // ɪ (sit)
    private const val IY = 51     // i (see) — ASCII 'i' at 51
    private const val AO = 76     // ɔ (law)
    private const val UH = 135    // ʊ (book)
    private const val UW = 63     // u (food) — ASCII 'u' at 63
    private const val UH_R = 83   // ər (her) — use schwa
    private const val OW = 57     // o (go) — ASCII 'o' at 57
    private const val AW = 69     // aʊ → use AA + W approximation
    private const val AY = 69     // aɪ → use AA + Y approximation
    private const val OY = 76     // ɔɪ → use AO + Y approximation
    private const val EY = 47     // eɪ → use ASCII 'e' at 47

    // ---- Stress markers ----
    private const val STRESS_PRIMARY = 156   // ˈ
    private const val STRESS_SECONDARY = 157 // ˌ

    /**
     * Convert English text to a token ID sequence ready for the model.
     *
     * Pipeline:
     *   1. Normalize (lowercase, expand common abbreviations, collapse whitespace)
     *   2. Word-by-word g2p → IPA token IDs
     *   3. Intersperse pad (0) between every token (add_blank=true)
     *   4. Prepend pad (the intersperse convention is [0, t1, 0, t2, 0, ...])
     *
     * @param text English input text.
     * @return IntArray of token IDs, already interspersed with pad.
     */
    fun textToTokenIds(text: String): IntArray {
        val normalized = normalize(text)
        val tokens = mutableListOf<Int>()

        // Word-by-word processing. Each word gets a primary stress on its
        // first vowel (rough approximation of eSpeak's stress assignment).
        val words = normalized.split(' ').filter { it.isNotEmpty() }
        for ((wordIdx, word) in words.withIndex()) {
            val wordTokens = g2pWord(word)
            if (wordTokens.isNotEmpty()) {
                // Add primary stress before the first vowel of the word
                // (only for content words with 2+ phonemes).
                if (wordTokens.size >= 2 && hasVowel(wordTokens)) {
                    tokens.add(STRESS_PRIMARY)
                }
                tokens.addAll(wordTokens)
            }
            // Space between words (but not after the last word)
            if (wordIdx < words.size - 1) {
                tokens.add(SPACE)
            }
        }

        // Intersperse pad (0) between every token: [0, t1, 0, t2, 0, ...]
        // This matches commons.intersperse(sequence, 0) which prepends 0
        // and inserts 0 between each element.
        val interspersed = IntArray(tokens.size * 2 + 1)
        interspersed[0] = PAD
        for (i in tokens.indices) {
            interspersed[i * 2 + 1] = tokens[i]
            interspersed[i * 2 + 2] = PAD
        }
        return interspersed
    }

    private fun hasVowel(tokens: List<Int>): Boolean {
        val vowels = setOf(AA, AE, AH, EH, IH, IY, AO, UH, UW, OW, EY, AW, AY, OY)
        return tokens.any { it in vowels }
    }

    /**
     * Normalize text: lowercase, expand common abbreviations and numbers,
     * collapse whitespace, strip unsupported characters.
     */
    private fun normalize(text: String): String {
        var s = text.lowercase().trim()
        // Common abbreviations
        s = s.replace(Regex("\\bmr\\."), "mister")
        s = s.replace(Regex("\\bmrs\\."), "missus")
        s = s.replace(Regex("\\bms\\."), "miss")
        s = s.replace(Regex("\\bdr\\."), "doctor")
        s = s.replace(Regex("\\bprof\\."), "professor")
        s = s.replace(Regex("\\bst\\."), "saint")
        s = s.replace(Regex("\\bvs\\."), "versus")
        s = s.replace(Regex("\\betc\\."), "et cetera")
        s = s.replace(Regex("\\be\\.g\\."), "for example")
        s = s.replace(Regex("\\bi\\.e\\."), "that is")
        // Numbers 0-20 → words (basic)
        s = s.replace(Regex("\\b0\\b"), "zero")
        s = s.replace(Regex("\\b1\\b"), "one")
        s = s.replace(Regex("\\b2\\b"), "two")
        s = s.replace(Regex("\\b3\\b"), "three")
        s = s.replace(Regex("\\b4\\b"), "four")
        s = s.replace(Regex("\\b5\\b"), "five")
        s = s.replace(Regex("\\b6\\b"), "six")
        s = s.replace(Regex("\\b7\\b"), "seven")
        s = s.replace(Regex("\\b8\\b"), "eight")
        s = s.replace(Regex("\\b9\\b"), "nine")
        s = s.replace(Regex("\\b10\\b"), "ten")
        // Strip anything that's not a-z, space, or basic punctuation
        s = s.replace(Regex("[^a-z .,!?;:']"), " ")
        s = s.replace(Regex("\\s+"), " ")
        return s.trim()
    }

    /**
     * Convert a single word (no spaces) to a list of IPA token IDs.
     * Uses rule-based consonant/vowel matching.
     */
    private fun g2pWord(word: String): List<Int> {
        if (word.isEmpty()) return emptyList()

        val tokens = mutableListOf<Int>()
        val w = word.trim('\'', '-')
        val chars = w.toCharArray()
        var i = 0

        while (i < chars.size) {
            val c = chars[i]
            val next = if (i + 1 < chars.size) chars[i + 1] else 0.toChar()
            val next2 = if (i + 2 < chars.size) chars[i + 2] else 0.toChar()

            // ---- Punctuation (handled at word boundaries) ----
            when (c) {
                ',' -> { tokens.add(PUNCT_COMMA); i++; continue }
                '.' -> { tokens.add(PUNCT_PERIOD); i++; continue }
                '!' -> { tokens.add(PUNCT_BANG); i++; continue }
                '?' -> { tokens.add(PUNCT_QMARK); i++; continue }
                ';' -> { tokens.add(PUNCT_SEMI); i++; continue }
                ':' -> { tokens.add(PUNCT_COLON); i++; continue }
                '\'', '-' -> { i++; continue }  // skip apostrophes/hyphens
            }

            // ---- Digraphs (check two-letter combos first) ----
            val digraph = processDigraph(c, next, next2)
            if (digraph != null) {
                tokens.addAll(digraph)
                i += if (isConsonantDigraph(c, next)) 2 else 1
                continue
            }

            // ---- Single char ----
            val single = processSingle(c, next)
            if (single != null) {
                tokens.addAll(single)
                i++
                continue
            }

            // Unknown char — skip
            i++
        }
        return tokens
    }

    /**
     * Process two/three-letter digraphs: th, sh, ch, ph, wh, ng, ck, etc.
     * Returns the token list and how many chars to consume (via the
     * isConsonantDigraph helper).
     */
    private fun processDigraph(c: Char, next: Char, next2: Char): List<Int>? {
        val pair = "$c$next"
        return when (pair) {
            // th → θ (voiceless) or ð (voiced, e.g. "the", "this")
            "th" -> {
                // Voiced 'th' at start of common function words
                val wordStart = true  // simplified
                if (listOf("the", "this", "that", "them", "then", "there", "their", "they", "thus", "although").any { it.startsWith("th") }) {
                    listOf(DH)
                } else {
                    listOf(TH)
                }
            }
            "sh" -> listOf(SH)
            "ch" -> listOf(SH)  // ch → ʃ (close enough)
            "ph" -> listOf(F)
            "wh" -> listOf(W)
            "ng" -> listOf(NG)  // word-final ng → ŋ
            "ck" -> listOf(K)
            "qu" -> listOf(K, W)
            "gh" -> {
                // 'gh' silent after vowels (e.g. "high", "though"), else f
                if (next2 == 0.toChar()) emptyList() else listOf(F)
            }
            else -> null
        }
    }

    private fun isConsonantDigraph(c: Char, next: Char): Boolean {
        val pair = "$c$next"
        return pair in setOf("th", "sh", "ch", "ph", "wh", "ng", "ck", "qu", "gh")
    }

    /**
     * Process a single character (vowel or consonant) with context.
     */
    private fun processSingle(c: Char, next: Char): List<Int>? {
        return when (c) {
            // ---- Consonants (mostly 1:1) ----
            'p' -> listOf(P)
            'b' -> listOf(B)
            't' -> listOf(T)
            'd' -> listOf(D)
            'k' -> listOf(K)
            'g' -> listOf(G)
            'f' -> listOf(F)
            'v' -> listOf(V)
            's' -> listOf(S)
            'z' -> listOf(Z)
            'h' -> listOf(H)
            'm' -> listOf(M)
            'n' -> listOf(N)
            'l' -> listOf(L)
            'r' -> listOf(R)
            'w' -> listOf(W)
            'y' -> listOf(Y)
            'j' -> listOf(JH)
            'x' -> listOf(K, S)  // x → ks
            'q' -> listOf(K)     // q (without u) → k

            // ---- Vowels (context-dependent) ----
            'a' -> listOf(vowelA(next))
            'e' -> listOf(vowelE(next))
            'i' -> listOf(vowelI(next))
            'o' -> listOf(vowelO(next))
            'u' -> listOf(vowelU(next))

            else -> null
        }
    }

    // ---- Vowel context rules (simplified) ----
    private fun vowelA(next: Char): Int {
        return when {
            next == 'r' -> AH        // ar → ər (car)
            next == 'i' || next == 'y' -> EY  // ai/ay → eɪ (rain, say)
            next == 'u' -> AO        // au → ɔ (caught)
            next == 'e' -> EY        // ae → eɪ (rarely)
            next == 'o' -> AO        // ao → ɔ
            next == 'l' || next == 'w' -> AO  // all, law → ɔ
            next == 's' || next == 'f' || next == 'n' || next == 'th' -> AE  // ask, after, ankle (a-fronting)
            next == 0.toChar() -> AH  // word-final a → schwa
            else -> AE                // default: cat
        }
    }

    private fun vowelE(next: Char): Int {
        return when {
            next == 'r' -> UH_R      // er → ər (her)
            next == 'e' -> IY        // ee → i (see)
            next == 'a' -> EY        // ea → eɪ (great) — simplified
            next == 'i' -> IY        // ei → i (receive) — simplified
            next == 'w' -> UW        // ew → u (few)
            next == 0.toChar() -> AH  // word-final silent e → skip (but use schwa)
            else -> EH                // bed
        }
    }

    private fun vowelI(next: Char): Int {
        return when {
            next == 'r' -> UH_R      // ir → ər (bird)
            next == 'e' || next == 'y' -> AY  // ie/iy → aɪ (pie, fly)
            next == 'o' -> AY        // io → aɪ (lion) — simplified
            next == 0.toChar() -> AY  // word-final i → aɪ (hi)
            else -> IH                // sit
        }
    }

    private fun vowelO(next: Char): Int {
        return when {
            next == 'r' -> AO         // or → ɔ (for)
            next == 'o' -> UW         // oo → u (food)
            next == 'u' -> UW         // ou → u (soup) — simplified
            next == 'w' -> OW         // ow → oʊ (low)
            next == 'i' -> OY         // oi → ɔɪ (oil)
            next == 'y' -> OY         // oy → ɔɪ (boy)
            next == 'a' -> OW         // oa → oʊ (boat) — simplified
            next == 0.toChar() -> OW  // word-final o → oʊ (go)
            else -> AA                // default: dog → ɑ (simplified from ɔ)
        }
    }

    private fun vowelU(next: Char): Int {
        return when {
            next == 'r' -> UH_R      // ur → ər (fur)
            next == 'e' -> UW        // ue → u (blue)
            next == 0.toChar() -> UW  // word-final u → u (flu)
            next == 'i' || next == 'y' -> UW  // ui/uy → u (juice)
            else -> UH                // book
        }
    }
}
