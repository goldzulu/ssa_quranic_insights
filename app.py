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

# Get app version from VERSION file
def get_app_version():
    try:
        with open("VERSION", "r") as version_file:
            return version_file.read().strip()
    except FileNotFoundError:
        return "Unknown"

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
        dataset_url = "https://raw.githubusercontent.com/goldzulu/ssa_quranic_insights/main/dataset/The_Quran_Dataset.csv"
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
    """Format an ayah with enhanced styling for Arabic text and complete reference information"""
    # Safely access DataFrame columns with fallbacks for missing columns
    ayah_ar = ayah.get('ayah_ar', ayah.get('arabic_text', 'Arabic text not available'))
    ayah_en = ayah.get('ayah_en', ayah.get('english_translation', 'Translation not available'))
    
    # Get surah information
    surah_name_en = ayah.get('surah_name_en', 'Unknown')
    surah_name_roman = ayah.get('surah_name_roman', '')
    surah_no = ayah.get('surah_no', 'Unknown')
    
    # Format surah name with both transliteration and English translation
    if surah_name_roman and surah_name_en:
        surah_display = f"{surah_name_roman} ({surah_name_en})"
    elif surah_name_roman:
        surah_display = surah_name_roman
    elif surah_name_en:
        surah_display = surah_name_en
    else:
        surah_display = f"Surah {surah_no}"
    
    # Get ayah numbers
    ayah_no_surah = ayah.get('ayah_no_surah', 'Unknown')
    ayah_no_quran = ayah.get('ayah_no_quran', 'Unknown')
    
    # Add Juz information
    juz_no = ayah.get('juz_no', '')
    juz_info = f", Juz: {juz_no}" if juz_no else ""
    
    # Complete reference information
    reference_text = f"Surah: {surah_display} ({surah_no}), Verse: {ayah_no_surah}{juz_info}"
    
    # Create a formatted reference with styled Arabic text
    return f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <div style="text-align: right; font-size: 36px; font-family: 'Traditional Arabic', 'Scheherazade', 'Amiri', serif; line-height: 1.8; margin-bottom: 20px; color: #000; direction: rtl; padding: 15px; background-color: #fcfcfc; border-radius: 8px;">
            {ayah_ar}
        </div>
        <div style="font-size: 16px; margin-bottom: 15px; font-style: italic; color: #444; line-height: 1.5;">
            {ayah_en}
        </div>
        <div style="font-size: 14px; color: #666; border-top: 1px solid #e0e0e0; padding-top: 12px; margin-top: 5px;">
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
    found_match = False
    for col in ['surah_name_en', 'surah_name_roman']:
        if col in df.columns:
            # Exact match first
            mask = df[col].str.lower() == surah_name_lower
            if mask.any():
                found_match = True
                break
                
            # Try without 'al-' prefix
            mask = df[col].str.lower() == clean_name
            if mask.any():
                found_match = True
                break
                
            # Try with contains
            mask = df[col].str.lower().str.contains(clean_name)
            if mask.any():
                found_match = True
                break
    
    # If no match found
    if not found_match or not mask.any():
        return f"Could not find any surah matching '{surah_name}'. Please check the spelling or try a different name."
    
    # Get the matched verses
    results = df[mask].copy()
    if results.empty:
        return f"No verses found for surah '{surah_name}'."
    
    # Get surah information
    surah_info = results.iloc[0]
    surah_no = surah_info.get('surah_no', 'Unknown')
    surah_name_en = surah_info.get('surah_name_en', '')
    surah_name_roman = surah_info.get('surah_name_roman', '')
    
    # Format surah name with both transliteration and English translation
    if surah_name_roman and surah_name_en:
        surah_display = f"{surah_name_roman} ({surah_name_en})"
    elif surah_name_roman:
        surah_display = surah_name_roman
    elif surah_name_en:
        surah_display = surah_name_en
    else:
        surah_display = f"Surah {surah_no}"
    
    verse_count = len(results)
    
    # Return summary and verses with enhanced formatting
    response = f"<h3>Found {verse_count} verses in Surah {surah_display} (#{surah_no})</h3>"
    
    # Add the first 10 verses or all if less than 10
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
    
    # Get surah information
    surah_info = results.iloc[0]
    surah_name_en = surah_info.get('surah_name_en', '')
    surah_name_roman = surah_info.get('surah_name_roman', '')
    
    # Format surah name with both transliteration and English translation
    if surah_name_roman and surah_name_en:
        surah_display = f"{surah_name_roman} ({surah_name_en})"
    elif surah_name_roman:
        surah_display = surah_name_roman
    elif surah_name_en:
        surah_display = surah_name_en
    else:
        surah_display = f"Surah {surah_number}"
    
    verse_count = len(results)
    
    # Return summary and verses with enhanced formatting
    response = f"<h3>Found {verse_count} verses in Surah {surah_display} (#{surah_number})</h3>"
    
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
        found_match = False
        for col in ['surah_name_en', 'surah_name_roman']:
            if col in df.columns:
                surah_filter = df[col].str.lower().str.contains(clean_name)
                if surah_filter.any():
                    found_match = True
                    break
        
        if not found_match or not surah_filter.any():
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
    
    # Get surah information
    surah_no = ayah.get('surah_no', 'Unknown')
    surah_name_en = ayah.get('surah_name_en', '')
    surah_name_roman = ayah.get('surah_name_roman', '')
    
    # Format surah name with both transliteration and English translation
    if surah_name_roman and surah_name_en:
        surah_display = f"{surah_name_roman} ({surah_name_en})"
    elif surah_name_roman:
        surah_display = surah_name_roman
    elif surah_name_en:
        surah_display = surah_name_en
    else:
        surah_display = f"Surah {surah_no}"
    
    response = f"<h3>Verse {ayah_number} from Surah {surah_display} (#{surah_no})</h3>"
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
    
    # Return summary and sample verses with enhanced formatting
    response = f"<h3>Found {verse_count} verses in {juz_range_text}</h3>"
    
    # Group by surah and get counts
    if 'surah_name_en' in results.columns:
        surah_counts = results.groupby(['surah_no', 'surah_name_en']).size()
        response += "<div style='margin-bottom: 20px;'><h4>Verses by Surah:</h4><ul>"
        for (surah_no, surah_name), count in surah_counts.items():
            response += f"<li><strong>Surah {surah_name} ({surah_no})</strong>: {count} verses</li>"
        response += "</ul></div>"
    
    response += "<h4>Sample verses:</h4>"
    
    # Show one verse from each of the first 3 surahs
    shown_surahs = set()
    sample_verses = []
    
    for _, ayah in results.iterrows():
        surah_no = ayah.get('surah_no')
        if surah_no not in shown_surahs and len(shown_surahs) < 3:
            shown_surahs.add(surah_no)
            sample_verses.append(ayah)
    
    # Display the sample verses with enhanced formatting
    for ayah in sample_verses:
        response += format_ayah_reference(ayah)
    
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
        columns_to_get = ['surah_no', 'surah_name_en', 'surah_name_ar', 'surah_name_roman']
        available_columns = [col for col in columns_to_get if col in df.columns]
        unique_surahs = df[available_columns].drop_duplicates().sort_values('surah_no')
        surah_list = unique_surahs.values.tolist()
    elif 'surah_no' in df.columns:
        unique_surahs = df[['surah_no']].drop_duplicates().sort_values('surah_no')
        surah_list = unique_surahs.values.tolist()
    else:
        return "Could not find surah information in the dataset."
    
    response = "<h3>List of Surahs in the Quran</h3>"
    response += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;'>"
    
    for surah in surah_list:
        if len(surah) > 1:
            surah_no = surah[0]
            surah_name_en = surah[1] if len(surah) > 1 else ""
            surah_name_ar = surah[2] if len(surah) > 2 else ""
            surah_name_roman = surah[3] if len(surah) > 3 else ""
            
            # Format surah name with both transliteration and English translation
            if surah_name_roman and surah_name_en:
                surah_display = f"{surah_name_roman} ({surah_name_en})"
            elif surah_name_roman:
                surah_display = surah_name_roman
            elif surah_name_en:
                surah_display = surah_name_en
            else:
                surah_display = f"Surah {surah_no}"
            
            response += f"""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e0e0e0;'>
                <div style='font-weight: bold;'>{surah_no}. {surah_display}</div>
                <div style='font-size: 18px; text-align: right; direction: rtl;'>{surah_name_ar}</div>
            </div>
            """
        else:
            response += f"<div>Surah #{surah[0]}</div>"
    
    response += "</div>"
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

IMPORTANT: For every ayah (verse) displayed, always ensure that:
1. The original Arabic text is shown
2. The English translation is provided
3. The complete reference is included with:
   - Surah name in both transliteration and English translation (e.g., "Al-Fatiha (The Opener)")
   - Surah number
   - Ayah number

If you're unsure about the exact spelling of a surah name, use list_available_surahs to see the correct names.

Respond in a respectful, informative manner appropriate for discussing religious texts.
         
When responding about a particular surah, and only a few verses are in the response, make sure the user knows that it is an excerpt from the surah."""),
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

# Display version number in the footer
version = get_app_version()
st.sidebar.markdown(f"<div style='position: fixed; bottom: 10px; left: 20px; color: #888;'>Version {version}</div>", unsafe_allow_html=True)

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
