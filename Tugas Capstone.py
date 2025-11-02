# CapstoneGUI_AuraPutriMahabah_2210312024.py
# dashboard banner, quick stats, NLP analysis, and debug logs for import.
# Author: Aura Putri Mahabah (2210312024)
# Run: python CapstoneGUI_AuraPutriMahabah_2210312024.py
# Requires: pandas, nltk, textblob, scikit-learn
# Install: pip install pandas nltk textblob scikit-learn

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import sqlite3, csv, threading, re, os, sys
import pandas as pd

# NLP libs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources (download if missing)
_nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'vader_lexicon': 'sentiment/vader_lexicon'
}
for res_name, res_path in _nltk_resources.items():
    try:
        nltk.data.find(res_path)
    except Exception:
        try:
            nltk.download(res_name, quiet=True)
        except Exception:
            pass  # proceed even if download fails

# ---------------- Colors and theme ----------------
BG_MAIN = "#FFFFFF"
BANNER_BG = "#A4C3B2"
BANNER_TEXT = "#FFFFFF"
CARD_BG = "#F6FFF8"
ACCENT = "#3B5249"
BTN_PRIMARY = "#A4C3B2"
BTN_SECOND = "#DDEEE6"

# ---------------- NLP Processor ----------------
class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            self.sia = None
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words and len(w) > 2]
        return ' '.join(tokens)

    def extract_keywords(self, text, top_n=10):
        processed = self.preprocess_text(text)
        if not processed:
            return []
        try:
            tfidf = self.vectorizer.fit_transform([processed])
            features = self.vectorizer.get_feature_names_out()
            scores = tfidf.toarray()[0]
            kw_scores = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
            return [kw for kw, s in kw_scores[:top_n] if s > 0]
        except Exception:
            return []

    def analyze_sentiment(self, text):
        if not isinstance(text, str) or not text.strip():
            return {'neg':0.0,'neu':0.0,'pos':0.0,'compound':0.0}
        if self.sia:
            try:
                return self.sia.polarity_scores(text)
            except Exception:
                pass
        try:
            tb = TextBlob(text)
            pol = tb.sentiment.polarity
            if pol > 0.05:
                pos, neg, neu = pol, 0.0, 1 - pol
            elif pol < -0.05:
                neg, pos, neu = -pol, 0.0, 1 + pol
            else:
                pos = neg = 0.0; neu = 1.0
            return {'neg':neg,'neu':neu,'pos':pos,'compound':pol}
        except Exception:
            return {'neg':0.0,'neu':0.0,'pos':0.0,'compound':0.0}

    def calculate_similarity(self, t1, t2):
        p1 = self.preprocess_text(t1)
        p2 = self.preprocess_text(t2)
        if not p1 or not p2:
            return 0.0
        try:
            tfidf = self.vectorizer.fit_transform([p1,p2])
            if tfidf.shape[0] < 2:
                return 0.0
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            return 0.0

# ---------------- Main GUI ----------------
class CapstoneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Capstone Project GUI - NLP-Based Research Paper Analysis (Aura Putri Mahabah - 2210312024)")
        self.root.geometry("1200x820")
        self.root.configure(bg=BG_MAIN)

        # DB setup
        self.db_path = os.path.join(os.path.dirname(__file__), 'database-2210312024.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

        # NLP
        self.nlp = NLPProcessor()

        # Style
        self.setup_style()

        # Banner + quick stats
        self.banner_frame = tk.Frame(self.root, bg=BANNER_BG)
        self.banner_frame.pack(fill="x", side="top")
        self.setup_banner()

        self.quickstats_frame = tk.Frame(self.root, bg=BG_MAIN, pady=8)
        self.quickstats_frame.pack(fill="x", padx=12, pady=(8,0))
        self.setup_quickstats()

        # Notebook tabs
        self.notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)

        # create tabs
        self.info_tab = ttk.Frame(self.notebook)
        self.create_tab = ttk.Frame(self.notebook)
        self.read_tab = ttk.Frame(self.notebook)
        self.update_tab = ttk.Frame(self.notebook)
        self.import_tab = ttk.Frame(self.notebook)
        self.ai_tab = ttk.Frame(self.notebook)
        self.delete_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.info_tab, text="Project Info")
        self.notebook.add(self.create_tab, text="Create/Insert")
        self.notebook.add(self.read_tab, text="View/Search")
        self.notebook.add(self.update_tab, text="Update")
        self.notebook.add(self.import_tab, text="Import Excel/CSV")
        self.notebook.add(self.ai_tab, text="NLP Analysis")
        self.notebook.add(self.delete_tab, text="Delete")

        # Setup each tab
        self.setup_info_tab()
        self.setup_create_tab()
        self.setup_read_tab()
        self.setup_update_tab()
        self.setup_import_tab()
        self.setup_nlp_tab()
        self.setup_delete_tab()

        # Initial load
        self.load_all_data()
        self.update_quickstats()

    def setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("Custom.TNotebook.Tab", background=BTN_SECOND, padding=(12,6), font=("Segoe UI",10))
        style.map("Custom.TNotebook.Tab", background=[("selected", BTN_PRIMARY)])
        style.configure("TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        style.configure("TLabel", background=BG_MAIN, foreground=ACCENT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BANNER_BG, foreground=BANNER_TEXT, font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=BANNER_BG, foreground=BANNER_TEXT, font=("Segoe UI", 10))
        style.configure("TButton", background=BTN_PRIMARY, foreground=BANNER_TEXT, font=("Segoe UI", 10))

    def setup_banner(self):
        title = tk.Label(self.banner_frame, text="NLP-Based Research Paper Analysis", bg=BANNER_BG, fg=BANNER_TEXT, font=("Segoe UI", 20, "bold"))
        title.pack(side="left", padx=20, pady=14)
        subtitle = tk.Label(self.banner_frame, text="Capstone Project by Aura Putri Mahabah (2210312024)", bg=BANNER_BG, fg=BANNER_TEXT, font=("Segoe UI", 10))
        subtitle.pack(side="left", padx=10, pady=18)

    def setup_quickstats(self):
        card_container = tk.Frame(self.quickstats_frame, bg=BG_MAIN)
        card_container.pack(fill="x", padx=6)
        self.stat_total = tk.Frame(card_container, bg=CARD_BG, bd=0)
        self.stat_total.pack(side="left", padx=6, ipadx=12, ipady=10, fill="both", expand=True)
        tk.Label(self.stat_total, text="Total Papers", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(6,2))
        self.total_val = tk.Label(self.stat_total, text="0", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 18, "bold"))
        self.total_val.pack(anchor="w", padx=8, pady=(0,6))

        self.stat_sent = tk.Frame(card_container, bg=CARD_BG, bd=0)
        self.stat_sent.pack(side="left", padx=6, ipadx=12, ipady=10, fill="both", expand=True)
        tk.Label(self.stat_sent, text="Rata-rata Sentiment (Compound)", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(6,2))
        self.sent_val = tk.Label(self.stat_sent, text="0.0000", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 16, "bold"))
        self.sent_val.pack(anchor="w", padx=8, pady=(0,6))

        self.stat_year = tk.Frame(card_container, bg=CARD_BG, bd=0)
        self.stat_year.pack(side="left", padx=6, ipadx=12, ipady=10, fill="both", expand=True)
        tk.Label(self.stat_year, text="Tahun Terbaru", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(6,2))
        self.year_val = tk.Label(self.stat_year, text="N/A", bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 18, "bold"))
        self.year_val.pack(anchor="w", padx=8, pady=(0,6))

    def update_quickstats(self):
        try:
            self.cursor.execute("SELECT COUNT(*) FROM research_papers")
            total = self.cursor.fetchone()[0] or 0
            self.total_val.config(text=str(total))

            self.cursor.execute("SELECT AVG(sentiment_score) FROM research_papers WHERE sentiment_score IS NOT NULL")
            avg_sent = self.cursor.fetchone()[0]
            avg_sent_display = f"{avg_sent:.4f}" if avg_sent is not None else "0.0000"
            self.sent_val.config(text=avg_sent_display)

            self.cursor.execute("SELECT MAX(year) FROM research_papers WHERE year IS NOT NULL AND year != ''")
            latest = self.cursor.fetchone()[0]
            self.year_val.config(text=str(latest) if latest else "N/A")
        except Exception as e:
            print("Quickstats update error:", e)
            self.total_val.config(text="0"); self.sent_val.config(text="0.0000"); self.year_val.config(text="N/A")

    # ---------------- Info tab ----------------
    def setup_info_tab(self):
        frame = ttk.Frame(self.info_tab, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Deskripsi Singkat", style="TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0,6))
        overview = (
            "Aplikasi desktop ini memungkinkan pengguna untuk menyimpan data penelitian (paper),\n"
            "mengimpor dataset CSV/Excel (termasuk format Zotero/Mendeley), serta melakukan analisis NLP.\n\n"
            "Petunjuk singkat:\n"
            "1. Gunakan tab 'Import Excel/CSV' untuk memasukkan dataset.\n"
            "2. Gunakan tab 'View/Search' untuk melihat/menelusuri data.\n"
            "3. Gunakan tab 'NLP Analysis' untuk menganalisis paper atau database.\n"
        )
        ttk.Label(frame, text=overview, wraplength=980, justify="left").pack(anchor="w")

    # ---------------- Create tab ----------------
    def setup_create_tab(self):
        canvas = tk.Canvas(self.create_tab, bg=BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.create_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_vars = {}
        fields = [
            ("Type","type","entry"),("Title","title","entry"),("Author 1","author1","entry"),
            ("Author 2","author2","entry"),("Author 3","author3","entry"),("Year","year","entry"),
            ("Rank","rank","combobox"),("Abstract","abstract","text"),("Introduction","introduction","text"),
            ("Literature","literature","text"),("Methodology","methodology","text"),("Experiment","experiment","text"),
            ("Discussion","discussion","text"),("Result","result","text"),("Conclusion","conclusion","text"),
            ("Reference","reference","text"),("NIM","nim","entry")
        ]
        r = 0
        for label_text, var_name, ftype in fields:
            ttk.Label(scroll_frame, text=label_text+":").grid(row=r, column=0, sticky="nw", padx=6, pady=6)
            if ftype == "entry":
                self.create_vars[var_name] = tk.StringVar()
                ttk.Entry(scroll_frame, textvariable=self.create_vars[var_name], width=72).grid(row=r, column=1, padx=6, pady=6, sticky="ew")
            elif ftype == "combobox":
                self.create_vars[var_name] = tk.StringVar()
                cb = ttk.Combobox(scroll_frame, textvariable=self.create_vars[var_name], width=70)
                cb['values'] = ['Q1','Q2','Q3','Sinta 1','Sinta 2','Sinta 3','Sinta 4','Sinta 5']
                cb.grid(row=r, column=1, padx=6, pady=6, sticky="ew")
            else:
                t = tk.Text(scroll_frame, height=4, width=72)
                t.grid(row=r, column=1, padx=6, pady=6, sticky="ew")
                self.create_vars[var_name] = t
            r += 1
        ttk.Button(scroll_frame, text="Insert Data", command=self.insert_data, style="TButton").grid(row=r, column=0, columnspan=2, pady=12)

    # ---------------- Read/Search tab ----------------
    def setup_read_tab(self):
        top_frame = ttk.Frame(self.read_tab)
        top_frame.pack(fill="x", padx=10, pady=6)
        ttk.Label(top_frame, text="Search by Title:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.search_var, width=42).pack(side="left", padx=8)
        ttk.Button(top_frame, text="Search", command=self.search_data).pack(side="left", padx=6)
        ttk.Button(top_frame, text="Show All", command=self.load_all_data).pack(side="left", padx=6)

        cols = ('ID','Type','Title','Author1','Author2','Author3','Year','Rank','NIM')
        self.tree = ttk.Treeview(self.read_tab, columns=cols, show='headings', height=18)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120 if c=='Title' else 90)
        vs = ttk.Scrollbar(self.read_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10,0), pady=6)
        vs.pack(side="right", fill="y", padx=(0,10), pady=6)
        self.tree.bind('<Double-1>', self.view_detail)

    # ---------------- Update tab ----------------
    def setup_update_tab(self):
        sf = ttk.Frame(self.update_tab)
        sf.pack(fill="x", padx=10, pady=6)
        ttk.Label(sf, text="Select ID to Update:").pack(side="left")
        self.update_id_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self.update_id_var, width=12).pack(side="left", padx=6)
        ttk.Button(sf, text="Load Data", command=self.load_update_data).pack(side="left", padx=6)

        canvas = tk.Canvas(self.update_tab, bg=BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.update_tab, orient="vertical", command=canvas.yview)
        self.update_frame = ttk.Frame(canvas)
        self.update_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.update_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ---------------- Import tab ----------------
    def setup_import_tab(self):
        frm = ttk.Frame(self.import_tab, padding=8)
        frm.pack(fill="both", expand=True)
        file_frame = ttk.LabelFrame(frm, text="Select File")
        file_frame.pack(fill="x", pady=(0,8))
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=90, state="readonly").pack(side="left", padx=6, pady=6)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="left", padx=6, pady=6)
        ttk.Button(file_frame, text="Preview Data", command=self.preview_data).pack(side="left", padx=6, pady=6)
        ttk.Button(file_frame, text="Import Data", command=self.import_data).pack(side="left", padx=6, pady=6)
        ttk.Button(file_frame, text="Download Template", command=self.download_template).pack(side="left", padx=6, pady=6)

        opt = ttk.LabelFrame(frm, text="Import Options")
        opt.pack(fill="x", pady=(0,8))
        ttk.Label(opt, text="Sheet Name (Excel):").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.sheet_var = tk.StringVar(value="Sheet1")
        ttk.Combobox(opt, textvariable=self.sheet_var, values=['Sheet1'], width=20).grid(row=0, column=1, padx=6, pady=3)
        self.has_header_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Baris pertama adalah header", variable=self.has_header_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3)

        preview = ttk.LabelFrame(frm, text="Data Preview (5 baris pertama)")
        preview.pack(fill="both", expand=True, pady=(0,8))
        preview_cols = ('Type','Title','Author','Publication Year','DOI','Abstract Note')
        self.preview_tree = ttk.Treeview(preview, columns=preview_cols, show='headings', height=6)
        for c in preview_cols:
            self.preview_tree.heading(c, text=c)
            self.preview_tree.column(c, width=160 if c=='Title' else 100)
        pv_scroll = ttk.Scrollbar(preview, orient="vertical", command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=pv_scroll.set)
        self.preview_tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        pv_scroll.pack(side="right", fill="y", padx=6, pady=6)

    def browse_file(self):
        ft = [("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        fn = filedialog.askopenfilename(title="Select Excel or CSV file", filetypes=ft)
        if fn:
            self.file_path_var.set(fn)

    def preview_data(self):
        fp = self.file_path_var.get()
        if not fp:
            messagebox.showerror("Error", "Pilih file dulu!")
            return
        try:
            if fp.lower().endswith('.csv'):
                df = pd.read_csv(fp)
            else:
                df = pd.read_excel(fp, sheet_name=self.sheet_var.get())
            df.columns = df.columns.str.strip()
            # clear preview tree
            for it in self.preview_tree.get_children(): self.preview_tree.delete(it)
            # select common columns to display
            for _, row in df.head(5).iterrows():
                vals = [
                    row.get('Item Type') or row.get('type') or '',
                    row.get('Title') or row.get('title') or '',
                    row.get('Author') or row.get('author') or '',
                    row.get('Publication Year') or row.get('year') or '',
                    row.get('DOI') or row.get('doi') or '',
                    row.get('Abstract Note') or row.get('abstract') or ''
                ]
                self.preview_tree.insert('', tk.END, values=[str(v)[:150] for v in vals])
            messagebox.showinfo("Preview", f"Menampilkan 5 baris pertama dari file ({len(df)} baris total)")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca file: {str(e)}")

    def import_data(self):
        fp = self.file_path_var.get()
        if not fp:
            messagebox.showerror("Error", "Pilih file dulu!")
            return
        if not messagebox.askyesno("Konfirmasi", "Data dari file akan dimasukkan ke database. Lanjutkan?"):
            return

        try:
            if fp.lower().endswith('.csv'):
                df = pd.read_csv(fp)
            else:
                df = pd.read_excel(fp, sheet_name=self.sheet_var.get())

            df.columns = df.columns.str.strip()
            # quick debug: show first 3 rows in console
            try:
                print("DEBUG: first 3 rows of the file:")
                print(df.head(3).to_dict(orient='records'))
            except Exception:
                pass

            # detect Zotero-like format
            is_zotero = any(col in df.columns for col in ['Title','Author','Publication Year'])

            # progress UI
            progress_win = tk.Toplevel(self.root)
            progress_win.title("Mengimpor Data...")
            ttk.Label(progress_win, text=f"Mengimpor data dari {os.path.basename(fp)}", font=("Segoe UI", 10, "bold")).pack(pady=(10, 5))
            progress = ttk.Progressbar(progress_win, length=400, mode="determinate")
            progress.pack(padx=20, pady=10)
            progress_label = ttk.Label(progress_win, text="0%", font=("Segoe UI", 9))
            progress_label.pack(pady=(0, 10))

            total_rows = len(df)
            inserted = 0; failed = 0

            for idx, row in df.iterrows():
                try:
                    if is_zotero:
                        title = str(row.get('Title') or row.get('title') or '').strip()
                        author1 = str(row.get('Author') or row.get('author') or '').strip()
                        year = row.get('Publication Year') or row.get('Year') or row.get('year') or ''
                        abstract = str(row.get('Abstract Note') or row.get('abstract') or '')
                        reference = str(row.get('DOI') or row.get('doi') or '')
                        type_paper = str(row.get('Item Type') or row.get('type') or '')
                    else:
                        title = str(row.get('title','') or '').strip()
                        author1 = str(row.get('author1','') or row.get('author','') or '').strip()
                        year = row.get('year','') or row.get('Publication Year','') or ''
                        abstract = str(row.get('abstract','') or '')
                        reference = str(row.get('reference','') or row.get('DOI','') or '') 
                        type_paper = str(row.get('type','') or '')

                    # validation
                    if not title or not author1:
                        print(f"SKIP row {idx}: missing title or author -> title='{title}' author='{author1}'")
                        failed += 1
                        continue

                    # normalize year to integer if possible
                    try:
                        year_int = int(float(year))
                    except Exception:
                        year_int = None

                    # Insert
                    self.cursor.execute('''INSERT INTO research_papers 
                        (type, title, author1, author2, author3, year, rank, abstract, introduction, literature, methodology, experiment, discussion, result, conclusion, reference, nim)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            type_paper, title, author1, 
                            str(row.get('author2','') or ''), str(row.get('author3','') or ''),
                            year_int, str(row.get('rank','') or ''), abstract,
                            str(row.get('introduction','') or ''), str(row.get('literature','') or ''),
                            str(row.get('methodology','') or ''), str(row.get('experiment','') or ''),
                            str(row.get('discussion','') or ''), str(row.get('result','') or ''),
                            str(row.get('conclusion','') or ''), reference, "2210312024"
                        )
                    )
                    inserted += 1
                except Exception as e:
                    print(f"❌ Gagal insert baris {idx}: {e}", file=sys.stderr)
                    failed += 1
                    continue

                # update progress
                progress['value'] = ((idx + 1) / total_rows) * 100
                progress_label.config(text=f"{int(progress['value'])}%")
                progress_win.update_idletasks()

            # commit
            self.conn.commit()
            self.load_all_data(); self.update_quickstats(); self.load_delete_data()
            progress_win.destroy()

            messagebox.showinfo("Import Selesai", f"✅ Import selesai!\n\nBerhasil: {inserted} baris\nGagal: {failed} baris\nTotal: {total_rows} baris")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengimpor data: {str(e)}")

    def download_template(self):
        fn = filedialog.asksaveasfilename(title="Save template as", defaultextension=".xlsx", filetypes=[("Excel files","*.xlsx"),("CSV files","*.csv")])
        if not fn: return
        try:
            data = {'type':['Journal Article'],'title':['Sample Title'],'author1':['John Doe'],'year':[2024],'rank':['Q1'],'abstract':['Sample abstract'],'nim':['2210312024']}
            df = pd.DataFrame(data)
            if fn.lower().endswith('.xlsx'): df.to_excel(fn,index=False)
            else: df.to_csv(fn,index=False)
            messagebox.showinfo("Success", f"Template tersimpan di: {fn}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saat menyimpan template: {str(e)}")

    # ---------------- NLP tab ----------------
    def setup_nlp_tab(self):
        f = ttk.Frame(self.ai_tab, padding=8)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Analisis Paper (ID):").grid(row=0,column=0,sticky="w",padx=6,pady=6)
        self.nlp_id_var = tk.StringVar(); ttk.Entry(f,textvariable=self.nlp_id_var,width=12).grid(row=0,column=1,padx=6,pady=6)
        ttk.Button(f,text="Analyze Single Paper",command=self.analyze_single_paper).grid(row=0,column=2,padx=6,pady=6)
        ttk.Label(f, text="Compare dengan ID:").grid(row=1,column=0,sticky="w",padx=6,pady=6)
        self.compare_id_var = tk.StringVar(); ttk.Entry(f,textvariable=self.compare_id_var,width=12).grid(row=1,column=1,padx=6,pady=6)
        ttk.Button(f,text="Compare Papers",command=self.compare_papers).grid(row=1,column=2,padx=6,pady=6)

        ttk.Label(f, text="Database-wide NLP Analysis:").grid(row=2,column=0,columnspan=3,sticky="w",padx=6,pady=(12,6))
        ttk.Button(f, text="Summarize Database (NLP)", command=self.summarize_database_nlp).grid(row=3,column=0,padx=6,pady=6)
        ttk.Button(f, text="Analyze Trends (NLP)", command=self.analyze_trends_nlp).grid(row=3,column=1,padx=6,pady=6)
        ttk.Button(f, text="Author Analysis (NLP)", command=self.analyze_authors_nlp).grid(row=3,column=2,padx=6,pady=6)
        ttk.Button(f, text="Topic Clustering (NLP)", command=self.cluster_topics_nlp).grid(row=4,column=0,padx=6,pady=6)
        ttk.Button(f, text="Export All NLP Results to CSV", command=self.export_nlp_results).grid(row=4,column=1,padx=6,pady=6)
        ttk.Button(f, text="Save Insights", command=self.save_insight_to_file).grid(row=4,column=2,padx=6,pady=6)

        res_frame = ttk.LabelFrame(f, text="NLP Analysis Results", style="Card.TFrame")
        res_frame.grid(row=5,column=0,columnspan=3,sticky="nsew",padx=6,pady=8)
        self.nlp_output = scrolledtext.ScrolledText(res_frame, wrap=tk.WORD, height=18, bg=CARD_BG, relief="flat")
        self.nlp_output.pack(fill="both", expand=True, padx=6, pady=6)
        f.grid_rowconfigure(5, weight=1); f.grid_columnconfigure(2, weight=1)

    def analyze_single_paper(self):
        pid = self.nlp_id_var.get().strip()
        if not pid:
            messagebox.showerror("Error", "Masukkan ID yang valid")
            return
        try:
            self.cursor.execute("SELECT abstract,introduction,literature,methodology,title FROM research_papers WHERE id = ?", (pid,))
            row = self.cursor.fetchone()
            if not row:
                messagebox.showerror("Error", "Paper tidak ditemukan")
                return
            combined = ' '.join([r for r in row if r])
            if not combined.strip():
                self.nlp_output.delete("1.0", tk.END); self.nlp_output.insert(tk.END, f"Tidak ada konten untuk Paper ID {pid}\n"); return
            kws = self.nlp.extract_keywords(combined, top_n=10)
            sent = self.nlp.analyze_sentiment(combined)
            proc = self.nlp.preprocess_text(combined); wc = len(proc.split())
            # update DB
            try:
                self.cursor.execute("UPDATE research_papers SET keywords=?, sentiment_score=?, word_count=?, processed_text=? WHERE id=?", (', '.join(kws), sent.get('compound',0.0), wc, proc, pid))
                self.conn.commit()
            except Exception:
                pass
            # display
            self.nlp_output.delete("1.0", tk.END)
            self.nlp_output.insert(tk.END, f"--- NLP Analysis: Paper ID {pid} ---\n")
            self.nlp_output.insert(tk.END, f"Judul: {row[4] if len(row)>4 else 'N/A'}\n")
            self.nlp_output.insert(tk.END, f"Kata Kunci: {', '.join(kws)}\n")
            self.nlp_output.insert(tk.END, f"Sentiment (Compound): {sent.get('compound',0.0):.4f} (pos:{sent.get('pos',0.0):.2f}, neu:{sent.get('neu',0.0):.2f}, neg:{sent.get('neg',0.0):.2f})\n")
            self.nlp_output.insert(tk.END, f"Word Count (Processed): {wc}\n\nSample Processed Text:\n{proc[:600]}...\n")
            self.nlp_output.see(tk.END)
            self.update_quickstats()
        except Exception as e:
            messagebox.showerror("Error", f"Error saat analisis: {str(e)}")

    def compare_papers(self):
        id1 = self.nlp_id_var.get().strip(); id2 = self.compare_id_var.get().strip()
        if not id1 or not id2:
            messagebox.showerror("Error", "Masukkan kedua ID untuk perbandingan")
            return
        try:
            self.cursor.execute("SELECT abstract FROM research_papers WHERE id=?", (id1,)); r1=self.cursor.fetchone()
            self.cursor.execute("SELECT abstract FROM research_papers WHERE id=?", (id2,)); r2=self.cursor.fetchone()
            if not r1 or not r2:
                messagebox.showerror("Error", "Satu atau kedua paper tidak ditemukan"); return
            t1=r1[0] or ""; t2=r2[0] or ""
            if not t1.strip() or not t2.strip():
                self.nlp_output.insert(tk.END, "Konten tidak cukup untuk dibandingkan\n"); return
            sim = self.nlp.calculate_similarity(t1,t2)
            self.nlp_output.insert(tk.END, f"\n--- Similarity ---\nSimilarity antara ID {id1} dan {id2}: {sim:.4f}\n\n"); self.nlp_output.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Error saat perbandingan: {str(e)}")

    def summarize_database_nlp(self):
        self.nlp_output.insert(tk.END, "--- Membuat Ringkasan Database (NLP) ---\n"); self.nlp_output.see(tk.END)
        def task():
            try:
                self.cursor.execute("SELECT title,abstract FROM research_papers WHERE abstract IS NOT NULL AND abstract!=''"); papers=self.cursor.fetchall()
                if not papers:
                    self.nlp_output.insert(tk.END, "Tidak ada abstract untuk diringkas\n"); return
                all_abs = " ".join([p[1] for p in papers if p[1]])
                top_kw = self.nlp.extract_keywords(all_abs, top_n=20)
                overall_sent = self.nlp.analyze_sentiment(all_abs)
                self.cursor.execute("SELECT type, COUNT(*) FROM research_papers GROUP BY type"); type_cnt=self.cursor.fetchall()
                self.cursor.execute("SELECT rank, COUNT(*) FROM research_papers GROUP BY rank"); rank_cnt=self.cursor.fetchall()
                summary = f"Total Papers: {len(papers)}\nTop Keywords: {', '.join(top_kw)}\nOverall Sentiment (Compound): {overall_sent.get('compound',0.0):.4f}\n"
                summary += "Type Distribution:\n"
                for t,c in type_cnt: summary += f" - {t if t else 'N/A'}: {c}\n"
                summary += "Rank Distribution:\n"
                for r,c in rank_cnt: summary += f" - {r if r else 'N/A'}: {c}\n"
                try:
                    blob = TextBlob(all_abs); simple_summary = ' '.join(blob.sentences[:5])
                except Exception:
                    simple_summary = all_abs[:500] + ("..." if len(all_abs)>500 else "")
                self.nlp_output.insert(tk.END, f"\nDatabase Overview:\n{summary}\nSimple Summary:\n{simple_summary}\n\n"); self.nlp_output.see(tk.END)
            except Exception as e:
                self.nlp_output.insert(tk.END, f"Error membuat ringkasan: {str(e)}\n\n"); self.nlp_output.see(tk.END)
        threading.Thread(target=task, daemon=True).start()

    def analyze_trends_nlp(self):
        self.nlp_output.insert(tk.END, "--- Menganalisis Tren Penelitian (NLP) ---\n"); self.nlp_output.see(tk.END)
        def task():
            try:
                self.cursor.execute("SELECT year, GROUP_CONCAT(abstract || ' ' || title, ' ') FROM research_papers WHERE year IS NOT NULL AND year != '' GROUP BY year ORDER BY year")
                data=self.cursor.fetchall()
                if not data: self.nlp_output.insert(tk.END, "Tidak ada data tahunan\n"); return
                for year, combined in data:
                    kws=self.nlp.extract_keywords(combined, top_n=5); sent=self.nlp.analyze_sentiment(combined)
                    self.nlp_output.insert(tk.END, f"Tahun {year}:\n - Top Keywords: {', '.join(kws)}\n - Sentiment (Compound): {sent.get('compound',0.0):.4f}\n")
                self.nlp_output.insert(tk.END, "\n"); self.nlp_output.see(tk.END)
            except Exception as e:
                self.nlp_output.insert(tk.END, f"Error analisis tren: {str(e)}\n"); self.nlp_output.see(tk.END)
        threading.Thread(target=task, daemon=True).start()

    def analyze_authors_nlp(self):
        self.nlp_output.insert(tk.END, "--- Menganalisis Pola Author (NLP) ---\n"); self.nlp_output.see(tk.END)
        def task():
            try:
                self.cursor.execute("SELECT author1, GROUP_CONCAT(abstract || ' ' || title, ' ') FROM research_papers WHERE author1 IS NOT NULL AND author1 != '' GROUP BY author1")
                data=self.cursor.fetchall()
                if not data: self.nlp_output.insert(tk.END, "Tidak ada data author\n"); return
                for author, combined in data:
                    kws=self.nlp.extract_keywords(combined, top_n=5); sent=self.nlp.analyze_sentiment(combined)
                    self.nlp_output.insert(tk.END, f"Author: {author}\n - Top Areas: {', '.join(kws)}\n - Sentiment (Compound): {sent.get('compound',0.0):.4f}\n")
                self.nlp_output.insert(tk.END, "\n"); self.nlp_output.see(tk.END)
            except Exception as e:
                self.nlp_output.insert(tk.END, f"Error authors analysis: {str(e)}\n"); self.nlp_output.see(tk.END)
        threading.Thread(target=task, daemon=True).start()

    def cluster_topics_nlp(self):
        self.nlp_output.insert(tk.END, "--- Clustering Topik (heuristik) ---\n"); self.nlp_output.see(tk.END)
        def task():
            try:
                self.cursor.execute("SELECT id,title,abstract FROM research_papers WHERE abstract IS NOT NULL AND abstract != ''")
                papers=self.cursor.fetchall()
                if not papers: self.nlp_output.insert(tk.END, "Tidak ada paper\n"); return
                corpus=[self.nlp.preprocess_text((p[2] or '') + ' ' + (p[1] or '')) for p in papers]
                overall_kw=self.nlp.extract_keywords(' '.join(corpus), top_n=15)
                self.nlp_output.insert(tk.END, f"Dominant Keywords: {', '.join(overall_kw)}\n\n")
                groups={}
                for pid,title,abstract in papers:
                    pkw=self.nlp.extract_keywords((abstract or '') + ' ' + (title or ''), top_n=3)
                    for kw in pkw:
                        if kw in overall_kw:
                            groups.setdefault(kw,[]).append(f"ID {pid}: {title[:60]}...")
                for kw,titles in groups.items():
                    self.nlp_output.insert(tk.END, f"--- Topik: {kw} ({len(titles)} papers) ---\n")
                    for t in titles[:5]: self.nlp_output.insert(tk.END, f" - {t}\n")
                    if len(titles)>5: self.nlp_output.insert(tk.END, " ...\n")
                    self.nlp_output.insert(tk.END, "\n")
                self.nlp_output.see(tk.END)
            except Exception as e:
                self.nlp_output.insert(tk.END, f"Error clustering topics: {str(e)}\n"); self.nlp_output.see(tk.END)
        threading.Thread(target=task, daemon=True).start()

    def export_nlp_results(self):
        fn = filedialog.asksaveasfilename(title="Export NLP Results", defaultextension=".csv", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not fn: return
        try:
            self.cursor.execute("SELECT id,title,keywords,sentiment_score,word_count FROM research_papers")
            rows=self.cursor.fetchall()
            with open(fn,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["ID","Title","Keywords","Sentiment Score","Word Count"])
                for r in rows: w.writerow(r)
            messagebox.showinfo("Export Complete", f"NLP results diekspor ke {fn}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saat export: {str(e)}")

    def save_insight_to_file(self):
        content = self.nlp_output.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "Tidak ada insight untuk disimpan.")
            return
        try:
            fn = "output_insights.txt"
            with open(fn,"w",encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Saved", f"Insights disimpan ke {os.path.abspath(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")

    # ---------------- CRUD helpers ----------------
    def insert_data(self):
        try:
            data={}
            for k,v in self.create_vars.items():
                if isinstance(v, tk.Text): data[k]=v.get("1.0",tk.END).strip()
                else: data[k]=v.get().strip()
            if not data.get('title') or not data.get('author1') or not data.get('year'):
                messagebox.showerror("Error","Title, Author1, dan Year wajib diisi!"); return
            self.cursor.execute('''INSERT INTO research_papers (type,title,author1,author2,author3,year,rank,abstract,introduction,literature,methodology,experiment,discussion,result,conclusion,reference,nim) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                                (data.get('type',''),data['title'],data['author1'],data.get('author2',''),data.get('author3',''),data['year'],data.get('rank',''),data.get('abstract',''),data.get('introduction',''),data.get('literature',''),data.get('methodology',''),data.get('experiment',''),data.get('discussion',''),data.get('result',''),data.get('conclusion',''),data.get('reference',''),data.get('nim','')))
            self.conn.commit()
            messagebox.showinfo("Success","Data berhasil disimpan!")
            self.clear_create_form(); self.load_all_data(); self.update_quickstats(); self.load_delete_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error saat menyimpan: {str(e)}")

    def clear_create_form(self):
        for k,v in self.create_vars.items():
            if isinstance(v, tk.Text): v.delete("1.0",tk.END)
            else: v.set("")

    def load_all_data(self):
        try:
            for it in self.tree.get_children(): self.tree.delete(it)
        except Exception:
            pass
        try:
            self.cursor.execute('SELECT id,type,title,author1,author2,author3,year,rank,nim FROM research_papers')
            rows=self.cursor.fetchall()
            for r in rows: self.tree.insert('',tk.END,values=r)
            self.update_quickstats()
        except Exception as e:
            print("Error load_all_data:",e)

    def search_data(self):
        term=self.search_var.get().strip()
        if not term: self.load_all_data(); return
        for it in self.tree.get_children(): self.tree.delete(it)
        try:
            q='SELECT id,type,title,author1,author2,author3,year,rank,nim FROM research_papers WHERE title LIKE ?'
            self.cursor.execute(q,(f'%{term}%',)); rows=self.cursor.fetchall()
            for r in rows: self.tree.insert('',tk.END,values=r)
        except Exception as e:
            messagebox.showerror("Error", f"Error saat search: {str(e)}")

    def view_detail(self,event):
        sel=self.tree.selection()
        if not sel: return
        item=self.tree.item(sel[0]); rid=item['values'][0]
        try:
            self.cursor.execute('SELECT * FROM research_papers WHERE id=?',(rid,)); rec=self.cursor.fetchone()
            if rec: self.show_detail_window(rec)
        except Exception as e:
            messagebox.showerror("Error", f"Error saat ambil detail: {str(e)}")

    def show_detail_window(self, record):
        win=tk.Toplevel(self.root); win.title(f"Detail - ID: {record[0]}"); win.geometry("880x600")
        tw=scrolledtext.ScrolledText(win, wrap=tk.WORD); tw.pack(fill="both",expand=True,padx=10,pady=10)
        fields=['ID','Type','Title','Author1','Author2','Author3','Year','Rank','Abstract','Introduction','Literature','Methodology','Experiment','Discussion','Result','Conclusion','Reference','NIM','Keywords','Sentiment Score','Word Count','Processed Text']
        for i, f in enumerate(fields):
            val = record[i] if i < len(record) else 'N/A'
            tw.insert(tk.END, f"{f}: {val if val else 'N/A'}\n\n")
        tw.config(state=tk.DISABLED)

    def load_update_data(self):
        rid=self.update_id_var.get().strip()
        if not rid: messagebox.showerror("Error","Silakan masukkan ID!"); return
        try:
            self.cursor.execute('SELECT * FROM research_papers WHERE id=?',(rid,)); rec=self.cursor.fetchone()
            if not rec: messagebox.showerror("Error","Record tidak ditemukan!"); return
            for w in self.update_frame.winfo_children(): w.destroy()
            self.update_vars={}
            fields=[("Type","type","entry",rec[1]),("Title","title","entry",rec[2]),("Author 1","author1","entry",rec[3]),("Author 2","author2","entry",rec[4]),("Author 3","author3","entry",rec[5]),("Year","year","entry",rec[6]),("Rank","rank","combobox",rec[7]),("Abstract","abstract","text",rec[8]),("Introduction","introduction","text",rec[9]),("Literature","literature","text",rec[10]),("Methodology","methodology","text",rec[11]),("Experiment","experiment","text",rec[12]),("Discussion","discussion","text",rec[13]),("Result","result","text",rec[14]),("Conclusion","conclusion","text",rec[15]),("Reference","reference","text",rec[16]),("NIM","nim","entry",rec[17] if len(rec)>17 else "")]
            r=0
            for lbl,var,ftype,val in fields:
                ttk.Label(self.update_frame, text=lbl+":").grid(row=r,column=0,sticky="nw",padx=6,pady=6)
                if ftype=="entry":
                    sv=tk.StringVar(value=val if val else ""); self.update_vars[var]=sv; ttk.Entry(self.update_frame,textvariable=sv,width=72).grid(row=r,column=1,padx=6,pady=6,sticky="ew")
                elif ftype=="combobox":
                    sv=tk.StringVar(value=val if val else ""); self.update_vars[var]=sv; cb=ttk.Combobox(self.update_frame,textvariable=sv,values=['Q1','Q2','Q3','Sinta 1','Sinta 2','Sinta 3','Sinta 4','Sinta 5'],width=70); cb.grid(row=r,column=1,padx=6,pady=6,sticky="ew")
                else:
                    t=tk.Text(self.update_frame,height=4,width=72); t.grid(row=r,column=1,padx=6,pady=6,sticky="ew"); self.update_vars[var]=t
                    if val: t.insert("1.0", val)
                r+=1
            ttk.Button(self.update_frame,text="Update Data",command=self.update_data).grid(row=r,column=0,columnspan=2,pady=12)
        except Exception as e:
            messagebox.showerror("Error", f"Error saat load update: {str(e)}")

    def update_data(self):
        try:
            rid=self.update_id_var.get().strip()
            if not rid: messagebox.showerror("Error","No ID selected!"); return
            data={}
            for k,v in self.update_vars.items():
                if isinstance(v, tk.Text): data[k]=v.get("1.0",tk.END).strip()
                else: data[k]=v.get().strip()
            self.cursor.execute('''UPDATE research_papers SET type=?,title=?,author1=?,author2=?,author3=?,year=?,rank=?,abstract=?,introduction=?,literature=?,methodology=?,experiment=?,discussion=?,result=?,conclusion=?,reference=?,nim=? WHERE id=?''',
                                (data.get('type',''),data.get('title',''),data.get('author1',''),data.get('author2',''),data.get('author3',''),data.get('year',''),data.get('rank',''),data.get('abstract',''),data.get('introduction',''),data.get('literature',''),data.get('methodology',''),data.get('experiment',''),data.get('discussion',''),data.get('result',''),data.get('conclusion',''),data.get('reference',''),data.get('nim',''),rid))
            self.conn.commit(); messagebox.showinfo("Success","Data berhasil diupdate!"); self.load_all_data(); self.update_quickstats(); self.load_delete_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error saat update: {str(e)}")

    def setup_delete_tab(self):
        f = ttk.Frame(self.delete_tab, padding=8); f.pack(fill="both", expand=True)
        ttk.Label(f, text="Delete by ID:").pack(pady=6)
        self.delete_id_var = tk.StringVar(); ttk.Entry(f, textvariable=self.delete_id_var, width=20).pack(pady=6)
        ttk.Button(f, text="Delete", command=self.delete_data).pack(pady=6)
        ttk.Label(f, text="Current Data:").pack(pady=(12,6))
        cols=('ID','Title','Author1','Year')
        self.delete_tree = ttk.Treeview(f, columns=cols, show='headings', height=10)
        for c in cols: self.delete_tree.heading(c, text=c); self.delete_tree.column(c, width=220 if c=='Title' else 120)
        self.delete_tree.pack(fill="both", expand=True, pady=6)
        self.load_delete_data()

    def load_delete_data(self):
        for it in self.delete_tree.get_children(): self.delete_tree.delete(it)
        try:
            self.cursor.execute('SELECT id,title,author1,year FROM research_papers')
            rows=self.cursor.fetchall()
            for r in rows: self.delete_tree.insert('',tk.END,values=r)
        except Exception as e:
            print("Error load_delete_data:",e)

    def delete_data(self):
        rid=self.delete_id_var.get().strip()
        if not rid: messagebox.showerror("Error","Silakan masukkan ID!"); return
        if not messagebox.askyesno("Confirm Delete", f"Apakah yakin ingin menghapus record ID {rid}?"): return
        try:
            self.cursor.execute('DELETE FROM research_papers WHERE id=?',(rid,))
            if self.cursor.rowcount>0:
                self.conn.commit(); messagebox.showinfo("Success","Data berhasil dihapus!"); self.delete_id_var.set(""); self.load_all_data(); self.update_quickstats(); self.load_delete_data()
            else:
                messagebox.showerror("Error","Record tidak ditemukan!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saat menghapus: {str(e)}")

    def create_table(self):
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS research_papers (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, title TEXT NOT NULL, author1 TEXT NOT NULL, author2 TEXT, author3 TEXT, year INTEGER, rank TEXT, abstract TEXT, introduction TEXT, literature TEXT, methodology TEXT, experiment TEXT, discussion TEXT, result TEXT, conclusion TEXT, reference TEXT, nim TEXT, keywords TEXT, sentiment_score REAL, word_count INTEGER, processed_text TEXT)''')
            self.conn.commit()
        except Exception as e:
            print("Error create_table:",e)

    def __del__(self):
        try:
            if hasattr(self,'conn'): self.conn.close()
        except Exception:
            pass

# Run the app
def main():
    root = tk.Tk()
    app = CapstoneGUI(root)
    def on_closing():
        if messagebox.askokcancel("Quit", "Keluar aplikasi?"):
            try:
                if hasattr(app,'conn'): app.conn.close()
            except Exception: pass
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
