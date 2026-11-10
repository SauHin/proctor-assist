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
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Opus 5.5 | Data analysis | Milestone 1: meminta AI mengecek fakta OEP sebelum dipakai, yaitu file webcam, format `gt.txt`, pemetaan "pakai HP" di paper P1, dan lisensi. | Used: temuan masuk ke D11 dan PLAN §4 (commit c2ac500). | PENDING: baca sendiri `READ_ME.txt` di `data/raw/oep/` dan bagian 3.1 paper P1, lalu bandingkan dengan D11 dan PLAN §4. |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Opus 5.5 | Coding | Milestone 1: meminta AI menulis script download OEP yang hanya mengambil webcam, `gt.txt`, dan `READ_ME.txt`, dengan manifest SHA-256 sesuai D8. | Used: `download.py`, `test_download.py`, dan `manifests/oep_raw.sha256` di-commit (c2ac500). | PENDING: jalankan sendiri `pytest` dan `python -m proctor_assist.download`, lalu pastikan ke-49 file cocok dengan manifest. |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Opus 5.5 | Documentation | Milestone 1: meminta AI mencatat D11, memperbarui PLAN §4, README, dan `CLAUDE.md`, lalu menyusun prompt sesi berikutnya. | Used: D11, PLAN §4, README, dan aturan kerja di-commit (c2ac500, e3e35bd), sedangkan aturan log AI dan prompt belum. | PENDING: baca diff c2ac500 dan e3e35bd serta aturan log AI yang baru, lalu putuskan apakah aturan itu dipertahankan. |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Fable 5.2 | Data analysis | Milestone 2: meminta AI mencari arti kode tipe `gt.txt` di paper OEP (Atoum et al. 2017) dan mencocokkannya dengan data. | Used: temuan masuk ke D12 (`docs/decisions.md`) dan PLAN §4 (commit 1bd508f). | PENDING: baca bagian IV paper OEP versi IEEE Xplore, buka contact sheet di `data/eda/frames/`, lalu bandingkan dengan D12, terutama arti kode 5 dan 6 yang masih berupa inferensi. |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Fable 5.2 | Coding | Milestone 2: meminta AI menulis parser `gt.txt` dan script EDA untuk statistik label, properti video, gambar, dan frame contoh. | Used: `labels.py`, `eda.py`, tesnya, dan `results/eda/` di-commit (ad60fa3, f0d4a74, 64baa23). | PENDING: jalankan sendiri `pytest` dan `python -m proctor_assist.eda`, lalu cocokkan beberapa angka di `results/eda/summary.md` dengan isi `gt.txt` (misalnya subject7 baris 14 dan jumlah baris subject1). |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Fable 5.2 | Experimental design | Milestone 2: meminta AI menyusun draf opsi pemetaan kelas dan pembersihan label (baris rusak, duplikat, overlap). | Used: owner memilih tiap opsi lewat pertanyaan pilihan ganda di chat, lalu AI mengetik D14 sesuai aturan D16. Opsi yang ditolak tetap ada di file PROPOSAL. | Dilaporkan owner di chat (2026-11-10): melihat `code5.jpg` (HP di telinga pada 5 frame atau lebih), membandingkan `code1.jpg` dengan `code6.jpg`, dan melihat lembar frame subject1, lalu memilih sendiri tiap opsi dan menerima rekap akhir. PENDING: baca bagian IV paper OEP versi IEEE Xplore dan tinjau ulang teks D14. |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Fable 5.2 | Experimental design | Milestone 3: meminta AI memberi opsi desain split subject untuk protokol A dan B, lalu menulis kode split dan tes leakage sesuai pilihan owner. | Used: `splits.py`, `splits/subjects.json`, dan tesnya di-commit (1f1bd26). Dua pilihan desain diputuskan owner lewat pilihan ganda (D15). | PENDING: jalankan sendiri `pytest`, buka `splits/subjects.json`, lalu cek dengan tangan bahwa tidak ada subject di dua bagian dan tiap test set punya tepat satu dari subject 1, 6, 9, 24. Tinjau juga dua pilihan AI di D15 (3 actor validasi, dan actor validasi yang punya kode 5). |
| TBD | 2026-11-10 | Claude Code (Claude Desktop), Claude Fable 5.2 | Coding | Milestone 3: meminta AI menerapkan keputusan D14 ke kode label per detik (pemetaan kelas, mask, perbaikan baris, urutan kelas). | Used: `labels.py` (`clean_labels`, `second_labels`) dan tesnya di-commit (09374fe). | PENDING: jalankan sendiri `pytest`, lalu cocokkan konstanta di `labels.py` dengan D14 satu per satu, dan cek hasil `second_labels` untuk subject1 detik 150 sampai 216 dengan tangan. |
