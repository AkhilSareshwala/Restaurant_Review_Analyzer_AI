import os
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

# Initialize Gemini components
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.environ['GOOGLE_API_KEY'])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ['GOOGLE_API_KEY'])

# Global variable to store vector store
vector_store = None

def analyze_reviews(reviews_text):
    # More explicit prompt for themes
    
    themes_prompt = f"""
    Analyze these restaurant reviews and identify the 5-7 most frequently mentioned themes.
    For each theme, provide:
    1. A descriptive title (in bold using **)
    2. A summary of the theme
    3. 2-3 representative quotes from reviews
    
    Format exactly like this example:
    - **Theme: Excellent Service**
      Summary: Many customers praised the attentive and knowledgeable staff
      Examples: "Our waiter was very helpful", "Staff went above and beyond"

    Reviews:
    {reviews_text}
    """ 
    
    
    # More structured complaints prompt
    complaints_prompt = f"""
    Extract ALL negative feedback from these reviews and categorize each:
    1. First identify if the review contains any complaint (ignore positive reviews)
    2. For each complaint:
       - Type: [Service/Food/Ambiance/Cleanliness/Other]
       - Exact quote from review
       - Specific issue mentioned
    
    Format as a markdown table with these exact headers:
    | Complaint Type | Customer Quote | Specific Issue |
    |----------------|----------------|----------------|
    
    Example row:
    | Service | "We had to ask multiple times for water" | Slow drink refills |

    Reviews:
    {reviews_text}
    """
    
    # More directive classification prompt
    classify_prompt = f"""
    Classify each review's PRIMARY focus (choose ONLY ONE per review):
    - Service: Comments about staff, speed, attentiveness
    - Food: Comments about taste, preparation, menu items
    - Ambiance: Comments about decor, noise, seating
    - Cleanliness: Comments about hygiene, maintenance
    - Value: Comments about pricing, portions, deals
    - Other: Doesn't fit above categories
    
    Format as a markdown table with these exact headers:
    | Review Excerpt | Classification | Reason |
    |----------------|----------------|--------|
    
    Example row:
    | "The pasta was perfectly cooked" | Food | Focuses on food quality |

    Reviews:
    {reviews_text}
    """
    
    
    try:
        themes_result = llm.invoke(themes_prompt)
        complaints_result = llm.invoke(complaints_prompt)
        classify_result = llm.invoke(classify_prompt)
        
        prepare_vector_store(reviews_text)
    

        # Add validation to ensure tables are properly formatted
        def validate_table(content, required_headers):
            if not any(header in content for header in required_headers):
                return "No relevant data found"
            return content
            
        return {
            "themes": themes_result.content,
            "complaints": validate_table(complaints_result.content, ["Complaint Type", "Customer Quote"]),
            "classification": validate_table(classify_result.content, ["Review Excerpt", "Classification"])
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "themes": "",
            "complaints": "",
            "classification": ""
        }
    
    
def prepare_vector_store(reviews_text):
    global vector_store
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(reviews_text)
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    if not vector_store:
        return jsonify({"error": "Please analyze reviews first"}), 400
    
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Retrieve relevant context
        docs = vector_store.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about restaurant reviews.
        Use only the provided context to answer the question. If you don't know, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Generate answer
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            if 'Review' not in df.columns:
                return jsonify({"error": "CSV must contain a 'Review' column"}), 400
                
            reviews_text = "\n".join([f"Review {i+1}: {row['Review']}" for i, row in df.iterrows()])
            analysis = analyze_reviews(reviews_text)
            return jsonify(analysis)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file"}), 400

if __name__ == '__main__':
    app.run(debug=True)