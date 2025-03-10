import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
import requests
from io import StringIO
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# Load environment variables for local development
load_dotenv()

# Get OpenAI API key from Streamlit secrets or environment variables
def get_openai_api_key():
    # First try to get from Streamlit secrets
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        return st.secrets.OPENAI_API_KEY
    # Fall back to environment variable for local development
    return os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client for direct calls
openai_client = OpenAI(api_key=get_openai_api_key())

# Dataset path
dataset_path = "The_Quran_Dataset.csv"

# Load or download the dataset
@st.cache_data
def load_quran_dataset():
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        # If local file not found, download from GitHub
        dataset_url = "https://raw.githubusercontent.com/reemamemon/Quranic_Insights/main/The_Quran_Dataset.csv"
        response = requests.get(dataset_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        # Save locally for future use
        df.to_csv(dataset_path, index=False)
    
    # Clean and convert column names to ensure consistent access
    # Convert all column names to lowercase for consistent access
    df.columns = df.columns.str.lower()
    return df

# Format the ayah reference for display
def format_ayah_reference(ayah):
    """Format an ayah with enhanced styling for Arabic text"""
    # Safely access DataFrame columns with fallbacks for missing columns
    ayah_ar = ayah.get('ayah_ar', ayah.get('arabic_text', 'Arabic text not available'))
    ayah_en = ayah.get('ayah_en', ayah.get('english_translation', 'Translation not available'))
    
    # Build reference string with only available columns
    ref_parts = []
    
    # Add Surah information
    if 'surah_name_en' in ayah:
        ref_parts.append(f"Surah: {ayah['surah_name_en']}")
    elif 'surah_name_roman' in ayah:
        ref_parts.append(f"Surah: {ayah['surah_name_roman']}")
    
    # Add Ayah number information
    if 'ayah_no_surah' in ayah:
        ref_parts.append(f"Verse: {ayah['ayah_no_surah']}")
    elif 'ayah_no_quran' in ayah:
        ref_parts.append(f"Quran verse: {ayah['ayah_no_quran']}")
    
    # Add Juz information
    if 'juz_no' in ayah:
        ref_parts.append(f"Juz: {ayah['juz_no']}")
    
    reference_text = ", ".join(ref_parts) if ref_parts else "Reference information not available"
    
    # Create a formatted reference with styled Arabic text
    return f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <div style="text-align: right; font-size: 28px; font-family: 'Traditional Arabic', 'Scheherazade', serif; line-height: 1.7; margin-bottom: 15px; color: #000;">
            {ayah_ar}
        </div>
        <div style="font-size: 16px; margin-bottom: 15px; font-style: italic; color: #444;">
            {ayah_en}
        </div>
        <div style="font-size: 14px; color: #666;">
            {reference_text}
        </div>
    </div>
    """

# Define tool schemas using Pydantic models
class SurahNameQuery(BaseModel):
    surah_name: str = Field(description="The name of the surah in English or Romanized form")

class SurahNumberQuery(BaseModel):
    surah_number: int = Field(description="The number of the surah (1-114)")

class AyahNumberQuery(BaseModel):
    surah: str = Field(description="The name or number of the surah")
    ayah_number: int = Field(description="The verse (ayah) number within the surah")

class JuzRangeQuery(BaseModel):
    start_juz: int = Field(description="Starting juz number (1-30)")
    end_juz: Optional[int] = Field(description="Ending juz number (if querying a range)")

class SemanticQuery(BaseModel):
    query: str = Field(description="The semantic query to search for relevant verses")

# Define LangChain tools
@tool(args_schema=SurahNameQuery)
def get_surah_by_name(surah_name: str) -> str:
    """
    Retrieve all verses from a surah by its name (in English or Romanized form).
    """
    df = load_quran_dataset()
    
    # Try different ways to match the surah name
    surah_name_lower = surah_name.lower()
    
    # Remove 'al-' prefix if present
    if surah_name_lower.startswith('al-'):
        clean_name = surah_name_lower[3:]
    else:
        clean_name = surah_name_lower
    
    # Try different column names and matching strategies
    for col in ['surah_name_en', 'surah_name_roman']:
        if col in df.columns:
            # Exact match first
            mask = df[col].str.lower() == surah_name_lower
            if mask.any():
                break
                
            # Try without 'al-' prefix
            mask = df[col].str.lower() == clean_name
            if mask.any():
                break
                
            # Try with contains
            mask = df[col].str.lower().str.contains(clean_name)
            if mask.any():
                break
    
    # If no match found
    if not mask.any():
        return f"Could not find any surah matching '{surah_name}'. Please check the spelling or try a different name."
    
    # Get the matched verses
    results = df[mask].copy()
    if results.empty:
        return f"No verses found for surah '{surah_name}'."
    
    surah_info = results.iloc[0]
    surah_actual_name = surah_info.get('surah_name_en', surah_name)
    verse_count = len(results)
    
    # Return summary and verses with enhanced formatting
    response = f"<h3>Found {verse_count} verses in Surah {surah_actual_name}</h3>"
    
    # Add the first 5 verses or all if less than 5
    for i, (_, ayah) in enumerate(results.head(min(10, len(results))).iterrows()):
        response += format_ayah_reference(ayah)
    
    if verse_count > 10:
        response += f"<p><em>... and {verse_count - 10} more verses.</em></p>"
    
    return response

@tool(args_schema=SurahNumberQuery)
def get_surah_by_number(surah_number: int) -> str:
    """
    Retrieve all verses from a surah by its number (1-114).
    """
    if not 1 <= surah_number <= 114:
        return "Invalid surah number. Please provide a number between 1 and 114."
    
    df = load_quran_dataset()
    
    # Try to find surah by number
    if 'surah_no' in df.columns:
        results = df[df['surah_no'] == surah_number].copy()
    else:
        return "Surah number column not found in the dataset."
    
    if results.empty:
        return f"No verses found for surah number {surah_number}."
    
    surah_name = results.iloc[0].get('surah_name_en', f"Surah {surah_number}")
    verse_count = len(results)
    
    # Return summary and verses with enhanced formatting
    response = f"<h3>Found {verse_count} verses in Surah {surah_name} (#{surah_number})</h3>"
    
    # Add the first 10 verses or all if less than 10
    for i, (_, ayah) in enumerate(results.head(min(10, len(results))).iterrows()):
        response += format_ayah_reference(ayah)
    
    if verse_count > 10:
        response += f"<p><em>... and {verse_count - 10} more verses.</em></p>"
    
    return response

@tool(args_schema=AyahNumberQuery)
def get_specific_ayah(surah: str, ayah_number: int) -> str:
    """
    Retrieve a specific verse (ayah) by surah name/number and verse number.
    """
    df = load_quran_dataset()
    
    # Check if surah is a number or name
    try:
        surah_number = int(surah)
        if 'surah_no' in df.columns:
            surah_filter = df['surah_no'] == surah_number
        else:
            return "Surah number column not found in the dataset."
    except ValueError:
        # It's a name
        surah_name_lower = surah.lower()
        
        # Remove 'al-' prefix if present
        if surah_name_lower.startswith('al-'):
            clean_name = surah_name_lower[3:]
        else:
            clean_name = surah_name_lower
        
        # Try different columns for surah name
        for col in ['surah_name_en', 'surah_name_roman']:
            if col in df.columns:
                surah_filter = df[col].str.lower().str.contains(clean_name)
                if surah_filter.any():
                    break
        
        if not surah_filter.any():
            return f"Could not find any surah matching '{surah}'. Please check the spelling or try using the surah number."
    
    # Filter by ayah number
    if 'ayah_no_surah' in df.columns:
        ayah_filter = df['ayah_no_surah'] == ayah_number
    else:
        return "Ayah number column not found in the dataset."
    
    # Apply both filters
    results = df[surah_filter & ayah_filter].copy()
    
    if results.empty:
        return f"No verse found for surah '{surah}' verse {ayah_number}."
    
    # There should be exactly one match
    ayah = results.iloc[0]
    surah_name = ayah.get('surah_name_en', surah)
    
    response = f"<h3>Verse {ayah_number} from Surah {surah_name}:</h3>"
    response += format_ayah_reference(ayah)
    
    return response

@tool(args_schema=JuzRangeQuery)
def get_juz_range(start_juz: int, end_juz: Optional[int] = None) -> str:
    """
    Retrieve verses from a specific juz or range of juz.
    """
    if not 1 <= start_juz <= 30:
        return "Invalid juz number. Please provide a number between 1 and 30."
    
    if end_juz is not None and not 1 <= end_juz <= 30:
        return "Invalid end juz number. Please provide a number between 1 and 30."
    
    if end_juz is not None and end_juz < start_juz:
        return "End juz number must be greater than or equal to start juz number."
    
    # If end_juz is not provided, just get the single juz
    if end_juz is None:
        end_juz = start_juz
    
    df = load_quran_dataset()
    
    # Check if juz column exists
    if 'juz_no' not in df.columns:
        return "Juz number column not found in the dataset."
    
    # Filter by juz range
    results = df[(df['juz_no'] >= start_juz) & (df['juz_no'] <= end_juz)].copy()
    
    if results.empty:
        return f"No verses found for juz {start_juz}" + (f" to {end_juz}" if end_juz != start_juz else "")
    
    verse_count = len(results)
    juz_range_text = f"Juz {start_juz}" + (f" to {end_juz}" if end_juz != start_juz else "")
    
    # Return summary and sample verses (first 5)
    response = f"Found {verse_count} verses in {juz_range_text}.\n\n"
    
    # Group by surah and get counts
    surah_counts = results.groupby('surah_name_en').size() if 'surah_name_en' in results.columns else results.groupby('surah_no').size()
    
    response += "Verses by Surah:\n"
    for surah, count in surah_counts.items():
        response += f"- {surah}: {count} verses\n"
    
    response += "\nSample verses:\n"
    # Add sample verses (first verse from first 3 surahs)
    shown_surahs = 0
    prev_surah = None
    
    for _, ayah in results.iterrows():
        current_surah = ayah.get('surah_name_en', ayah.get('surah_no', 'Unknown'))
        
        if current_surah != prev_surah and shown_surahs < 3:
            response += f"\nFrom Surah {current_surah}, Verse {ayah.get('ayah_no_surah', 'Unknown')}:\n"
            response += f"Translation: {ayah.get('ayah_en', 'Not available')}\n"
            
            prev_surah = current_surah
            shown_surahs += 1
            
        if shown_surahs >= 3:
            break
    
    return response

@tool(args_schema=SemanticQuery)
def search_semantically(query: str) -> str:
    """
    Search for verses semantically related to the query using OpenAI embeddings.
    """
    df = load_quran_dataset()
    
    # Generate embedding for the query
    embed_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = embed_response.data[0].embedding
    
    # Process all verses if not already processed
    cache_file = "quran_embeddings_cache.npz"
    
    if not os.path.exists(cache_file):
        # Create enriched texts for better semantic search
        if 'enriched_text' not in df.columns:
            df['enriched_text'] = df.apply(
                lambda row: f"Surah: {row.get('surah_name_en', '')} | Verse: {row.get('ayah_no_surah', '')} | {row.get('ayah_en', '')}", 
                axis=1
            )
        
        # Generate embeddings for all verses
        with st.spinner("Generating embeddings for verses (one-time operation)..."):
            all_texts = df['enriched_text' if 'enriched_text' in df.columns else 'ayah_en'].tolist()
            
            # Process in batches of 100
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i+batch_size]
                response = openai_client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            # Save to cache
            np.savez(cache_file, embeddings=all_embeddings)
    
    # Load cached embeddings
    embeddings_data = np.load(cache_file)
    all_embeddings = embeddings_data['embeddings']
    
    # Convert to numpy array for calculations
    all_embeddings = np.array(all_embeddings)
    query_embedding = np.array(query_embedding)
    
    # Calculate cosine similarity
    # Normalize embeddings
    all_embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Calculate similarities
    similarities = np.dot(all_embeddings_norm, query_embedding_norm)
    
    # Get top 5 most similar
    top_indices = np.argsort(-similarities)[:5]
    
    # Gather the retrieved Ayahs
    retrieved_ayahs = df.iloc[top_indices].copy()
    
    if retrieved_ayahs.empty:
        return "No semantically relevant verses found for your query."
    
    response = f"<h3>Found 5 verses semantically related to: '{query}'</h3>"
    
    # Add the retrieved verses with enhanced formatting
    for i, (_, ayah) in enumerate(retrieved_ayahs.iterrows(), 1):
        response += format_ayah_reference(ayah)
    
    return response

@tool
def list_available_surahs() -> str:
    """
    List all available surahs in the Quran dataset with their numbers and names.
    """
    df = load_quran_dataset()
    
    # Find unique surahs
    if 'surah_name_en' in df.columns and 'surah_no' in df.columns:
        unique_surahs = df[['surah_no', 'surah_name_en']].drop_duplicates().sort_values('surah_no')
        surah_list = unique_surahs.values.tolist()
    elif 'surah_no' in df.columns:
        unique_surahs = df[['surah_no']].drop_duplicates().sort_values('surah_no')
        surah_list = unique_surahs.values.tolist()
    else:
        return "Could not find surah information in the dataset."
    
    response = "List of Surahs in the Quran:\n\n"
    
    for surah in surah_list:
        if len(surah) > 1:
            response += f"{surah[0]}. {surah[1]}\n"
        else:
            response += f"Surah #{surah[0]}\n"
    
    return response

# Set up the LangChain agent
def setup_agent():
    # Define the tools
    tools = [
        get_surah_by_name,
        get_surah_by_number,
        get_specific_ayah,
        get_juz_range,
        search_semantically,
        list_available_surahs
    ]
    
    # Create the LLM with API key from secrets
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=get_openai_api_key()
    )
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Islamic AI assistant specializing in the Quran. Your goal is to help users find and understand verses from the Quran.

You have access to several tools that can help you retrieve Quranic verses in different ways:
1. Retrieve by surah name (get_surah_by_name)
2. Retrieve by surah number (get_surah_by_number)
3. Retrieve a specific verse by surah and verse number (get_specific_ayah)
4. Retrieve verses from a specific juz or range of juz (get_juz_range)
5. Search for verses semantically related to a query (search_semantically)
6. List all available surahs (list_available_surahs)

When a user asks about specific surahs like Al-Fatiha or Al-Ikhlas, use the appropriate tool to retrieve those surahs.
When a user asks about specific verses, use the specific verse retrieval tool.
When a user asks general questions about topics in the Quran, use the semantic search tool.

Always think carefully about which tool is most appropriate for the user's query. Be thorough and try to provide the most relevant Quranic verses.

If you're unsure about the exact spelling of a surah name, use list_available_surahs to see the correct names.

Respond in a respectful, informative manner appropriate for discussing religious texts.
         
When responding about a particular surah, and only a few verses are in the response, make sure the user know that it is an excerpt from the surah."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    return agent_executor

# Streamlit UI
st.title("Islamic AI Assistant: Quranic Insights")
st.write("Ask questions about the Quran, request specific surahs or verses, or explore topics from the Quran.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "<div" in message["content"]:
            # If the message contains HTML (from formatted ayahs)
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Initialize the agent (only once)
if "agent" not in st.session_state:
    st.session_state.agent = setup_agent()

# Input for user query
user_query = st.chat_input("Enter your question or request about the Quran:")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Run the agent
            with st.spinner("Searching the Quran..."):
                response = st.session_state.agent.invoke({"input": user_query})
                answer = response["output"]
            
            # Check if the answer contains HTML for formatted verses
            if "<div" in answer:
                # Update the placeholder with the response
                message_placeholder.markdown(answer, unsafe_allow_html=True)
            else:
                # Regular markdown for non-HTML content
                message_placeholder.markdown(answer)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
