# AI Usage Log: Proctor Assist (COMP6826001)

Log ini memenuhi Project Guideline §20 (AI Usage Log), §21 (AI and Experimental Integrity), dan §22 (AI Evaluation). Tabel di bawah akan disalin ke Technical Report bagian 10 (AI Usage Declaration).

## Aturan Penulisan

### Apa yang dicatat
- Catat setiap penggunaan AI yang **material**, yaitu yang memengaruhi kode, arsitektur, desain eksperimen, analisis, atau isi laporan. Koreksi typo atau pertanyaan umum yang tidak dipakai tidak perlu dicatat.
- Satu baris = satu tujuan penggunaan, bukan satu prompt. Satu sesi bisa menghasilkan 1–3 baris jika tujuannya berbeda (misalnya desain eksperimen dan debugging).
- Penggunaan Claude Code dicatat per milestone `docs/PLAN.md`, satu baris per tujuan (misalnya "Milestone 3: coding pipeline preprocessing").
- Jangan salin percakapan. Cukup ringkasan yang menunjukkan bagaimana AI dipakai (§20).
- Aplikasi tidak memakai layanan AI eksternal. Model pretrained yang dieksperimenkan (misalnya MobileNetV2 dan model video pretrained) adalah bagian dari sistem DL dan didokumentasikan di bab Model dan System Design, bukan di log ini.

### Isi tiap kolom
| Kolom | Isi | Contoh |
| :--- | :--- | :--- |
| **No.** | Nomor urut, tidak pernah dipakai ulang atau diurutkan ulang. | 3 |
| **Tanggal** | Tanggal sesi, format `YYYY-MM-DD`. | 2026-11-20 |
| **AI Tool** | Nama tool + model, bukan sekadar "LLM". | Claude Code (Claude Desktop) + Colab MCP |
| **Purpose** | Kategori: brainstorming, problem definition, system design, coding, debugging, experimental design, data analysis, documentation, writing/editing, UI development. | Debugging |
| **Prompt/Instruction Summary** | Satu kalimat tentang apa yang diminta. | Asked why val macro F1 drops after epoch 3 in E1 |
| **Output Used** | `Used` / `Partially used` / `Not used`, plus bagian mana yang dipakai. | Partially used: window sampler fix only |
| **Student Verification** | Tindakan konkret yang dilakukan, hasilnya, dan keputusan akhir. Lihat aturan di bawah. | Checked the subject IDs in every fold and found no subject in two splits; kept the fix, rejected the suggested lr change |

### Aturan kolom Student Verification
- Tulis **apa yang dilakukan + hasil/bukti + keputusan**. "Checked", "reviewed", atau "looks correct" tidak cukup.
- Jika menemukan kesalahan atau keterbatasan output AI, tulis di sini. Ini yang dinilai pada rubrik *AI Usage, Verification & Reflection* (10%).
- Hanya tulis verifikasi yang **sudah benar-benar dilakukan**. Jika belum, tulis `PENDING: <rencana verifikasi>` dan perbarui setelah dilakukan. Sebelum laporan dikumpulkan, tidak boleh ada `PENDING` yang tersisa.
- Pernyataan AI tidak pernah menjadi bukti verifikasi. Metric, hasil training, label ground truth, keputusan pemetaan kelas final, dan respons tester tidak boleh berasal dari AI (§21).

### Instruksi untuk AI saat diminta membuat entry dari suatu sesi
1. Baca sesi yang dimaksud, lalu tulis entry secara ringkas dan padat: satu kalimat per sel, dalam bahasa Indonesia.
2. Isi hanya yang terlihat jelas dari sesi. Jangan menebak tanggal, nomor urut, bagian output yang dipakai, atau verifikasi.
3. Jika ada detail yang hilang, **tanyakan langsung ke user dalam satu pesan**, biasanya:
   - nomor entry terakhir di log;
   - output mana yang dipakai, diubah, atau dibuang;
   - verifikasi apa yang sudah dilakukan beserta hasilnya (atau apakah ditandai `PENDING`).
4. Jangan pernah mengarang verifikasi, hasil tes, atau angka.

## AI Usage Log

| No. | Tanggal | AI Tool | Purpose | Prompt/Instruction Summary | Output Used | Student Verification |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- |
