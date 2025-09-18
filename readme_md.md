# QueryFi Assistant - Finance Expert 💰

An intelligent financial document analysis assistant powered by Streamlit and Ollama. QueryFi specializes in analyzing financial statements like Income Statements, Balance Sheets, and Cash Flow Statements.

## 🚀 Features

- **Document Upload**: Support for PDF and Excel files (up to 20MB)
- **Financial Statement Processing**: Automatic detection and parsing of Income Statements, Balance Sheets, and Cash Flow Statements
- **Intelligent Q&A**: Ask natural language questions about your financial documents
- **Quick Questions**: Pre-defined financial analysis questions for instant insights
- **Advanced Search**: TF-IDF vectorization with cosine similarity for relevant context retrieval
- **Safety Controls**: Built-in content safety checks and keyword filtering
- **Chat History**: Persistent conversation history with timestamps
- **System Parameters**: Adjustable temperature and max tokens for customized responses
- **Financial Metrics Extraction**: Automatic extraction of key financial metrics
- **Document Caching**: Optimized processing with Streamlit caching

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/queryfi-assistant.git
   cd queryfi-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama:**
   - Download Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Install the default model:
     ```bash
     ollama pull llama3.2
     ```
   - Start Ollama server:
     ```bash
     ollama serve
     ```

## 🚀 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload financial documents:**
   - Use the sidebar to upload PDF or Excel files
   - Supported formats: PDF, XLS, XLSX
   - Maximum file size: 20MB per file

3. **Ask questions:**
   - Use the quick questions for instant insights
   - Or type custom questions about your financial documents
   - Adjust system parameters (temperature, max tokens) as needed

## 📊 Supported Financial Documents

- **Income Statements**: Revenue, expenses, net income analysis
- **Balance Sheets**: Assets, liabilities, equity breakdown
- **Cash Flow Statements**: Operating, investing, financing activities
- **Financial Reports**: Multi-sheet Excel workbooks with financial data

## 🔧 Configuration

### Environment Variables

You can customize the application using environment variables:

```bash
# Ollama API configuration
export OLLAMA_API_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.2"
```

### System Parameters

- **Temperature**: Controls response creativity (0.0 - 1.0)
  - Lower values (0.0-0.5): More factual, precise responses
  - Higher values (0.6-1.0): More creative, interpretative insights
- **Max Tokens**: Controls response length (100 - 4000)

## 🔒 Safety Features

- Content safety checks with keyword filtering
- Professional conversation focus on financial topics
- Input sanitization and validation
- File size limits and processing timeouts

## 📚 Example Questions

- "What was the total revenue for the latest period?"
- "How did net income compare to the previous year?"
- "What were the major expense categories?"
- "What is the debt-to-equity ratio?"
- "What are the key financial highlights?"
- "What were the operating cash flows?"

## 🏗️ Architecture

```
QueryFi Assistant
├── Document Processing
│   ├── PDF text extraction (pdfplumber)
│   ├── Excel parsing (pandas)
│   └── Financial statement detection
├── Text Analysis
│   ├── TF-IDF vectorization
│   ├── Cosine similarity search
│   └── Context chunking
├── LLM Integration
│   ├── Ollama API communication
│   ├── Prompt engineering for finance
│   └── Response generation
└── User Interface
    ├── Streamlit web app
    ├── File upload interface
    └── Chat interface
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error:**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is installed: `ollama list`

2. **File Processing Errors:**
   - Ensure file size is under 20MB
   - Check file format (PDF, XLS, XLSX only)
   - Verify file is not corrupted

3. **Slow Processing:**
   - Large files may take time to process
   - Consider reducing max tokens for faster responses
   - Files with many sheets may be truncated for performance

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ollama](https://ollama.ai/)
- Uses [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF processing
- Financial analysis capabilities with pandas and scikit-learn