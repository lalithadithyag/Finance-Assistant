#all functionalities but comples and takes time to process documents
import os
import io
import re
import json
import requests
from typing import List, Tuple, Any, Dict, Optional
from datetime import datetime

import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Configuration
# ----------------------------
APP_TITLE = "QueryFi Assistant - Finance Expert"
MAX_UPLOAD_MB = 20
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Simple safety keywords
UNSAFE_KEYWORDS = ['bomb', 'weapon', 'kill', 'suicide', 'harm', 'hate', 'racist', 'porn', 'nude', 'sexual', 'bully', 'threat']

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="💰")
st.title("💰 " + APP_TITLE)
st.caption(f"LLM Provider: **Ollama** | Model: `{DEFAULT_MODEL}` | Hi! I'm your Finance Expert - Post any financial question here!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "financial_data" not in st.session_state:
    st.session_state.financial_data = {}
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1000

# ----------------------------
# Safety Functions
# ----------------------------
def is_content_safe(text: str) -> bool:
    """Simple safety check"""
    if not text:
        return True
    text_lower = text.lower()
    return not any(keyword in text_lower for keyword in UNSAFE_KEYWORDS)

def get_safety_message() -> str:
    return "Please keep our conversation professional and focused on financial topics. I'm here to help with your financial analysis needs."

# ----------------------------
# Enhanced Excel Processing for Financial Statements - FIXED VERSION
# ----------------------------
def clean_financial_value(value) -> Optional[str]:
    """Clean and standardize financial values"""
    if pd.isna(value) or value == "" or value is None:
        return None
    
    # Convert to string and clean
    str_val = str(value).strip()
    
    # Skip empty or very short values
    if len(str_val) < 1 or str_val in ['-', '—', 'N/A', 'n/a', 'NA']:
        return None
    
    return str_val

def format_financial_value(value):
    """More robust financial value formatting that handles ranges and complex notations"""
    if not value or pd.isna(value):
        return "--"
    
    str_val = str(value).strip()
    
    # Skip if it's just formatting characters
    if str_val in ['-', '—', 'N/A', 'n/a', 'NA', '']:
        return "--"
    
    # Handle ranges (e.g., "150 million to 127.5 million") - keep as-is
    if any(connector in str_val.lower() for connector in [' to ', ' - ', ' through ', ' vs ']):
        return str_val
    
    # Handle percentages
    if '%' in str_val:
        return str_val
    
    # Try to extract and format single numbers
    # Look for number patterns with optional units
    number_pattern = r'([\d,.-]+)\s*(million|billion|thousand|M|B|K|m|b|k)?'
    match = re.search(number_pattern, str_val, re.IGNORECASE)
    
    if match:
        try:
            num_str = match.group(1).replace(',', '')
            unit = match.group(2)
            
            # Handle negative numbers in parentheses
            if str_val.strip().startswith('(') and str_val.strip().endswith(')'):
                num_str = '-' + num_str.replace('-', '')
            
            num_val = float(num_str)
            
            # Apply unit multipliers
            if unit:
                unit_lower = unit.lower()
                if unit_lower in ['billion', 'b']:
                    num_val *= 1000000000
                elif unit_lower in ['million', 'm']:
                    num_val *= 1000000
                elif unit_lower in ['thousand', 'k']:
                    num_val *= 1000
            
            # Format the result
            if abs(num_val) >= 1000000000:
                return f"${num_val/1000000000:.1f}B"
            elif abs(num_val) >= 1000000:
                return f"${num_val/1000000:.1f}M"
            elif abs(num_val) >= 1000:
                return f"${num_val/1000:.1f}K"
            else:
                return f"${num_val:,.0f}"
                
        except (ValueError, AttributeError):
            pass
    
    # If all parsing fails, return the original value
    return str_val

def detect_financial_statement_type(df: pd.DataFrame, sheet_name: str) -> str:
    """Detect the type of financial statement based on content"""
    sheet_lower = sheet_name.lower()
    
    # Check sheet name first
    if any(term in sheet_lower for term in ['income', 'profit', 'loss', 'p&l', 'pnl']):
        return "Income Statement"
    elif any(term in sheet_lower for term in ['balance', 'position', 'assets', 'liabilities']):
        return "Balance Sheet"
    elif any(term in sheet_lower for term in ['cash', 'flow', 'cf']):
        return "Cash Flow Statement"
    
    # Check content
    content_str = ' '.join([str(col).lower() for col in df.columns if pd.notna(col)])
    if df.shape[0] > 0:
        first_col_content = ' '.join([str(val).lower() for val in df.iloc[:, 0].dropna() if str(val).strip()])
        content_str += ' ' + first_col_content
    
    # Income statement keywords
    income_keywords = ['revenue', 'sales', 'income', 'expenses', 'profit', 'loss', 'ebitda', 'operating income', 'net income']
    if any(keyword in content_str for keyword in income_keywords):
        return "Income Statement"
    
    # Balance sheet keywords
    balance_keywords = ['assets', 'liabilities', 'equity', 'cash and cash equivalents', 'accounts receivable', 'inventory']
    if any(keyword in content_str for keyword in balance_keywords):
        return "Balance Sheet"
    
    # Cash flow keywords
    cashflow_keywords = ['operating activities', 'investing activities', 'financing activities', 'cash flow']
    if any(keyword in content_str for keyword in cashflow_keywords):
        return "Cash Flow Statement"
    
    return "Financial Statement"

def process_financial_dataframe(df: pd.DataFrame, sheet_name: str) -> str:
    """Process a financial statement DataFrame into readable text - FIXED VERSION"""
    try:
        if df.empty:
            return f"Sheet '{sheet_name}' is empty.\n"
        
        # Detect statement type
        statement_type = detect_financial_statement_type(df, sheet_name)
        
        # Clean the dataframe
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        if df_clean.empty:
            return f"Sheet '{sheet_name}' contains no valid data.\n"
        
        text_parts = [f"\n=== {statement_type}: {sheet_name} ===\n"]
        
        # Try to identify the structure
        # Look for a column that might contain line items (usually first column)
        line_items_col = None
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Text column
                non_null_count = df_clean[col].notna().sum()
                if non_null_count > df_clean.shape[0] * 0.3:  # At least 30% non-null
                    line_items_col = col
                    break
        
        if line_items_col is None and df_clean.shape[1] > 0:
            line_items_col = df_clean.columns[0]
        
        # Process the data
        # Get column headers (periods/years)
        value_columns = [col for col in df_clean.columns if col != line_items_col]
        
        if value_columns:
            # Add header information
            header_text = "PERIODS: " + " | ".join([str(col) for col in value_columns]) + "\n\n"
            text_parts.append(header_text)
        
        # Process each row
        for idx, row in df_clean.iterrows():
            line_item = clean_financial_value(row.get(line_items_col, ''))
            if not line_item:
                continue
            
            # Skip header-like rows that are all caps or contain only formatting
            if line_item.isupper() and len(line_item) < 50:
                text_parts.append(f"\n--- {line_item} ---\n")
                continue
            
            # Build the line item text
            line_text = f"{line_item}:"
            
            # Add values for each period
            for col in value_columns:
                value = clean_financial_value(row.get(col, ''))
                formatted_value = format_financial_value(value)
                line_text += f" {formatted_value}"
            
            text_parts.append(line_text + "\n")
    
        return "".join(text_parts) + "\n"
        
    except Exception as e:
        # Fallback to simple table conversion
        try:
            return f"\n=== {sheet_name} (Fallback Format) ===\n{df.to_string(index=False, max_cols=10)}\n[Note: Used fallback formatting due to processing error: {str(e)}]\n"
        except:
            return f"\n=== {sheet_name} ===\nERROR: Could not process this sheet: {str(e)}\n"

def extract_text_from_excel_bytes(file_bytes: bytes, filename: str = "") -> str:
    """Enhanced Excel text extraction for financial statements - FIXED VERSION"""
    try:
        # Add timeout and error handling for Excel processing
        with st.spinner(f"Processing Excel file: {filename}"):
            # Try to read all sheets with error handling
            try:
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
            except Exception as e:
                return f"\n=== {filename} ===\nERROR: Could not read Excel file: {str(e)}\n"
            
            if not excel_data:
                return f"No sheets found in Excel file {filename}\n"
            
            all_text_parts = [f"\n=== EXCEL FILE: {filename} ===\n"]
            financial_summary = []
            
            # Limit processing to avoid hanging on large files
            sheet_count = 0
            max_sheets = 10  # Limit number of sheets to process
            
            for sheet_name, raw_df in excel_data.items():
                sheet_count += 1
                if sheet_count > max_sheets:
                    all_text_parts.append(f"\n[Note: Stopped processing after {max_sheets} sheets to prevent timeout]\n")
                    break
                    
                if raw_df.empty:
                    continue
                
                try:
                    # Try different header row positions (0, 1, 2) with timeout protection
                    best_df = None
                    best_score = 0
                    
                    for header_row in [0, 1, 2]:
                        try:
                            if header_row >= len(raw_df):
                                continue
                            
                            df_test = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=header_row)
                            
                            # Score this configuration
                            score = 0
                            if not df_test.empty:
                                # Prefer configurations with more numeric columns
                                numeric_cols = sum(1 for col in df_test.columns if df_test[col].dtype in ['int64', 'float64'])
                                score += numeric_cols * 2
                                
                                # Prefer configurations with meaningful column names
                                meaningful_cols = sum(1 for col in df_test.columns if isinstance(col, str) and len(str(col).strip()) > 2)
                                score += meaningful_cols
                                
                                if score > best_score:
                                    best_score = score
                                    best_df = df_test
                        
                        except Exception:
                            continue
                    
                    # Use the best configuration, or fallback to raw
                    df_to_process = best_df if best_df is not None else raw_df
                    
                    # Limit dataframe size to prevent hanging
                    if df_to_process.shape[0] > 1000:
                        df_to_process = df_to_process.head(1000)
                        all_text_parts.append(f"[Note: Sheet '{sheet_name}' truncated to first 1000 rows]\n")
                    
                    # Process this sheet
                    sheet_text = process_financial_dataframe(df_to_process, sheet_name)
                    all_text_parts.append(sheet_text)
                    
                    # Extract key metrics for summary
                    statement_type = detect_financial_statement_type(df_to_process, sheet_name)
                    financial_summary.append(f"- {statement_type} ({sheet_name}): {df_to_process.shape[0]} rows, {df_to_process.shape[1]} columns")
                    
                except Exception as e:
                    all_text_parts.append(f"\n=== {sheet_name} ===\nERROR: Could not process this sheet: {str(e)}\n")
                    continue
            
            # Add summary at the beginning
            if financial_summary:
                summary_text = "\nFILE SUMMARY:\n" + "\n".join(financial_summary) + "\n"
                all_text_parts.insert(1, summary_text)
            
            return "\n".join(all_text_parts)
    
    except Exception as e:
        error_msg = f"Error processing Excel file {filename}: {str(e)}"
        st.error(error_msg)
        return f"\n=== {filename} ===\nERROR: Could not process this Excel file: {str(e)}\n"

# ----------------------------
# Core Functions - FIXED VERSION
# ----------------------------
def sanitize_question(q: str, max_len: int = 2000) -> str:
    if not q:
        return ""
    q = re.sub(r"[\x00-\x1f\x7f]", " ", q).strip()
    return q[:max_len] + ("..." if len(q) > max_len else "")

def extract_text_from_pdf_bytes(file_bytes: bytes, filename: str = "") -> str:
    """Enhanced PDF text extraction with timeout protection"""
    try:
        with st.spinner(f"Processing PDF file: {filename}"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = ""
                page_count = 0
                max_pages = 50  # Limit pages to prevent hanging
                
                for page in pdf.pages:
                    page_count += 1
                    if page_count > max_pages:
                        text += f"\n[Note: Stopped processing after {max_pages} pages to prevent timeout]\n"
                        break
                        
                    try:
                        page_text = page.extract_text() or ""
                        # Try to extract tables as well
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                table_text = "\n".join(["\t".join([cell or "" for cell in row]) for row in table])
                                text += f"\n{table_text}\n"
                        text += page_text + "\n\n"
                    except Exception as e:
                        text += f"\n[Note: Error processing page {page_count}: {str(e)}]\n"
                        continue
                        
                return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text from {filename}: {e}")
        return f"ERROR: Could not process PDF {filename}: {str(e)}"

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Increased chunk size for better financial context"""
    if not text:
        return []
    words = text.split()
    chunks, step = [], max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

@st.cache_data(show_spinner=False)
def build_tfidf_index(chunks: List[str]) -> Tuple[TfidfVectorizer, Any]:
    """Build TF-IDF index with error handling"""
    if not chunks or len(chunks) == 0:
        return None, None
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", 
            max_features=25000,
            ngram_range=(1, 2),  # Include bigrams for better financial term matching
            min_df=1,
            max_df=0.95
        )
        X = vectorizer.fit_transform(chunks)
        return vectorizer, X
    except Exception as e:
        st.error(f"Error building search index: {str(e)}")
        return None, None

def retrieve_top_k(query: str, vectorizer: TfidfVectorizer, X, chunks: List[str], k: int = 5):
    """FIXED: Properly handle sparse matrix check"""
    # Check if any component is None or invalid
    if vectorizer is None or X is None or not chunks:
        return []
    
    try:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, X).flatten()
        top_idx = sims.argsort()[-k:][::-1]
        return [chunks[i] for i in top_idx if i < len(chunks) and sims[i] > 0.05]  # Lower threshold for financial docs
    except Exception as e:
        st.error(f"Error in document search: {str(e)}")
        return []

def extract_financial_metrics(text: str) -> Dict[str, str]:
    """Enhanced financial metrics extraction"""
    if not text:
        return {}
        
    metrics = {}
    
    # More comprehensive patterns for financial statements
    patterns = {
        'Total Revenue': [
            r'(?:total\s+)?(?:net\s+)?revenue[s]?\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?',
            r'(?:net\s+)?sales\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?'
        ],
        'Net Income': [
            r'net\s+(?:income|profit|earnings)\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?',
            r'(?:profit|income)\s+(?:after|net)\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?'
        ],
        'Total Assets': [
            r'total\s+assets\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?'
        ],
        'Total Equity': [
            r'(?:total\s+)?(?:shareholders?\'?\s+)?equity\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?'
        ]
    }
    
    for metric_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics[metric_name] = match.group(1)
                    break
            except Exception:
                continue
    
    return metrics

def query_ollama(prompt: str, model=DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Enhanced Ollama query with error handling and parameters"""
    try:
        payload = {
            "model": model, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)  # Increased timeout
        response.raise_for_status()
        j = response.json()
        return j.get("response", j.get("output", j.get("text", "[No response]")))
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)."
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try a shorter question."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"Error: {e}"

def add_to_chat_history(question: str, answer: str):
    """Add Q&A pair to chat history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "timestamp": timestamp
    })

def show_quick_questions():
    """Show predefined quick questions for financial documents"""
    st.subheader("🚀 Quick Questions")
    
    quick_questions = [
        "What was the total revenue for the latest period?",
        "What was the net income or profit?",
        "What were the major expenses categories?",
        "How did performance compare to previous year?",
        "What are the key financial highlights?",
        "What were the operating cash flows?",
        "What are the main assets and liabilities?",
        "What is the debt-to-equity ratio?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                return question
    return None

# ----------------------------
# Main Application - FIXED VERSION
# ----------------------------
def main():
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF or Excel files", 
            accept_multiple_files=True, 
            type=["pdf", "xls", "xlsx"],
            help="Upload financial statements like Income Statement, Balance Sheet, Cash Flow Statement"
        )
        
        # System Parameters Section
        st.divider()
        st.header("⚙️ System Parameters")
        
        # Temperature slider
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower values (0.0-0.5) for more factual responses, higher values (0.6-1.0) for more creative insights"
        )
        
        # Max tokens slider
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=st.session_state.max_tokens,
            step=100,
            help="Maximum length of the response. Higher values allow for more detailed answers."
        )
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.session_state.document_text = ""
            st.session_state.financial_data = {}
            st.rerun()
    
    # Process documents - FIXED VERSION
    if uploaded_files and not st.session_state.document_processed:
        # Show progress for large files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            all_text = []
            file_info = []
            
            for i, f in enumerate(uploaded_files):
                # Update progress
                progress = (i / len(uploaded_files))
                progress_bar.progress(progress)
                status_text.text(f"Processing {f.name}... ({i+1}/{len(uploaded_files)})")
                
                # File size check
                if f.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.warning(f"⚠️ File {f.name} is larger than {MAX_UPLOAD_MB}MB and may cause processing delays.")
                
                try:
                    raw = f.read()
                    file_info.append(f"📄 {f.name} ({f.size/1024:.1f} KB)")
                    
                    if f.name.lower().endswith(".pdf"):
                        extracted_text = extract_text_from_pdf_bytes(raw, f.name)
                    else:
                        extracted_text = extract_text_from_excel_bytes(raw, f.name)
                    
                    if extracted_text and not extracted_text.startswith("ERROR:"):
                        all_text.append(extracted_text)
                    else:
                        st.error(f"Failed to process {f.name}: {extracted_text}")
                        
                except Exception as e:
                    st.error(f"Error processing {f.name}: {str(e)}")
                    continue
            
            # Final processing
            progress_bar.progress(1.0)
            status_text.text("Finalizing document processing...")
            
            merged_text = "\n\n".join(all_text).strip()
            
            if merged_text:
                st.session_state.document_text = merged_text
                st.session_state.document_processed = True
                
                # Show file processing summary
                st.success(f"✅ Successfully processed {len(uploaded_files)} file(s):")
                for info in file_info:
                    st.success(info)
                
                # Extract key metrics
                try:
                    metrics = extract_financial_metrics(merged_text)
                    if metrics:
                        st.session_state.financial_data = metrics
                except Exception as e:
                    st.warning(f"Could not extract financial metrics: {str(e)}")
            else:
                st.error("❌ No text could be extracted from any of the uploaded files.")
                
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error during document processing: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            return

    if not st.session_state.document_processed:
        st.info("👆 Please upload PDF or Excel files containing financial statements to begin.")
        st.info("💡 **Supported Files:** Income Statements, Balance Sheets, Cash Flow Statements")
        return

    # Document preview (collapsible)
    with st.expander("📑 Document Preview", expanded=False):
        preview_text = st.session_state.document_text[:5000]  # Increased preview
        if len(st.session_state.document_text) > 5000:
            preview_text += "\n\n... (document continues)"
        st.text_area("Document Content", preview_text, height=300, disabled=True)

    # Build search index with better parameters for financial docs
    try:
        chunks = chunk_text(st.session_state.document_text, chunk_size=400, overlap=80)
        vectorizer, X = build_tfidf_index(chunks)
    except Exception as e:
        st.error(f"Error building search index: {str(e)}")
        vectorizer, X = None, None

    # Single column layout for chat interface
    st.subheader("💭 Ask Questions About Your Financial Documents")
    
    # Initialize session state for current question
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    
    # Check for quick question selection first
    quick_q = show_quick_questions()
    if quick_q and quick_q != st.session_state.current_question:
        st.session_state.current_question = quick_q
        st.rerun()
    
    # Question input with session state and Enter key support
    question = st.text_input(
        "Type your question here:", 
        value=st.session_state.current_question,
        key="question_input",
        placeholder="e.g., What was the revenue growth rate? or What were the main expenses?"
    )
    
    # Update session state when input changes
    if question != st.session_state.current_question:
        st.session_state.current_question = question
    
    question = sanitize_question(question)
    
    # Process question (both button click and Enter key)
    process_question = st.button("🚀 Ask", type="primary") or (question and question != st.session_state.get('last_processed_question', ''))
    
    if process_question and question:
        # Safety check
        if not is_content_safe(question):
            st.warning("⚠️ " + get_safety_message())
            return
        
        if vectorizer is None or X is None:
            st.warning("⚠️ Search functionality is limited due to document processing issues. Using basic text matching.")
            # Provide basic functionality even if search index failed
            context = st.session_state.document_text[:5000]  # Use first 5000 chars as context
        else:
            # Mark this question as processed to avoid reprocessing on rerun
            st.session_state.last_processed_question = question
            
            with st.spinner("🤔 Analyzing financial documents..."):
                # Retrieve relevant context
                top_chunks = retrieve_top_k(question, vectorizer, X, chunks, k=5)
                context = "\n\n".join(top_chunks)
        
        # Enhanced prompt for financial analysis
        history_context = ""
        if st.session_state.chat_history:
            recent_history = st.session_state.chat_history[-2:]  # Last 2 exchanges
            history_context = "\n".join([
                f"Previous Q: {h['question']}\nPrevious A: {h['answer'][:200]}...\n"
                for h in recent_history
            ])
        
        # Add friendly greeting if it's the first interaction
        greeting = ""
        if not st.session_state.chat_history:
            greeting = "Hi! I'm your Finance Assistant, specialized in analyzing financial documents and statements. "
        
        prompt = f"""{greeting}You are an expert financial analyst assistant. Answer the user's question using ONLY the provided financial document context.

{f"RECENT CONVERSATION CONTEXT: {history_context}" if history_context else ""}

FINANCIAL DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {question}

Instructions for Financial Analysis:
- Provide specific numbers, percentages, and financial metrics when available
- Format monetary amounts clearly (e.g., $1.2 million, $450K, 15.3%)
- If comparing periods, mention the specific years/quarters/periods
- For ratios or percentages, show the calculation when possible
- Identify trends (increasing, decreasing, stable) when comparing periods
- If the information isn't in the document, clearly state "This information is not available in the provided documents"
- Be precise and use financial terminology appropriately
- When discussing financial statements, specify which statement the data comes from
- Highlight any significant changes or notable figures

Answer:"""
        
        answer = query_ollama(prompt, DEFAULT_MODEL, st.session_state.temperature, st.session_state.max_tokens)
        
        # Add to chat history
        add_to_chat_history(question, answer)
        
        # Clear the current question
        st.session_state.current_question = ""
        st.rerun()
    
    # Display all chat history in chronological order (newest first)
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 Chat History")
        
        # Display in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"**🙋 You:** {chat['question']}")
                
                # Answer
                st.markdown(f"**🤖 Finance Assistant:** {chat['answer']}")
                
                # Timestamp
                st.caption(f"⏰ {chat['timestamp']}")
                
                # Add divider except for the last item
                if i < len(st.session_state.chat_history) - 1:
                    st.divider()

if __name__ == "__main__":
    main()