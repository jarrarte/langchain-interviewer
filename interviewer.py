import os
import json
from dotenv import load_dotenv
from typing import Optional


from TechnicalInterviewerApp import TechnicalInterviewerApp

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- Main Interviewer Application Class ---

def start_interview(interviewer : TechnicalInterviewerApp):
    """Starts the interactive interview loop."""
    
    print("\n--- Interview Started ---")
    
    greeting = "Hello!"
    if interviewer.candidate_name: 
        greeting = f"Hello {interviewer.candidate_name}!"
    print(f"Interviewer: {greeting} Welcome to the technical interview simulation.")
    
    print("Type 'stop', 'end interview', or 'finish' at any time to end.")
    print("You can also say 'talk about AI' or 'coding challenge' to switch topics.")
    
    while interviewer.interview_stage != "END":
        question = interviewer.generate_question()
        
        print(f"\nInterviewer: {question}")
        if interviewer.interview_stage == "END": 
            break
        
        user_answer = input("You: ")
        
        if user_answer.lower() in ["stop", "end interview", "finish"]: 
            interviewer.interview_stage = "END"
            print("\nInterviewer: Understood. Thank you for your time today!")
            break
        
        is_coding_challenge_question = (interviewer.interview_stage == "CODING_CHALLENGE" 
                                        and ("generate a" in question.lower() or "coding problem" in question.lower()) 
                                        and ("difficulty" in question.lower() or interviewer.coding_preferences))
        if interviewer.interview_stage != "CODING_SETUP" and not is_coding_challenge_question:
            feedback = interviewer._evaluate_answer(question, user_answer)
            print(f"\nInterviewer Feedback: {feedback}")
        elif is_coding_challenge_question: 
            print("\nInterviewer: Okay, thank you for providing your solution. I will evaluate it.")
        
        interviewer.transition_state(user_answer)
    print("\n--- Interview Ended ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration Loading ---
    CONFIG_FILE_PATH = "config.json"
    DEFAULT_GOOGLE_CHAT_MODEL = "gemini-2.5-pro-exp-03-25"
    DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
    DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
    DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    
    DEFAULT_RESUME_PATH = "placeholder_resume.pdf"
    DEFAULT_JOB_DESCRIPTION_PDF_PATH = "placeholder_job_description.pdf"   
    
    DEFAULT_LLM_PROVIDER = "google" # Default provider if not specified

    config = {}
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {CONFIG_FILE_PATH}")
    except FileNotFoundError:
        print(f"Warning: Configuration file '{CONFIG_FILE_PATH}' not found. Using default values.")
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{CONFIG_FILE_PATH}'. Using default values.")

    # --- Determine LLM Provider and Models ---
    llm_provider = config.get("llm_provider", DEFAULT_LLM_PROVIDER).lower()
    chat_model_name = None
    embedding_model_name = None

    if llm_provider == "google":
        chat_model_name = config.get("google_chat_model", DEFAULT_GOOGLE_CHAT_MODEL)
        embedding_model_name = config.get("google_embedding_model", DEFAULT_GOOGLE_EMBEDDING_MODEL)
        print(f"Using Google provider with Chat Model: {chat_model_name}, Embedding Model: {embedding_model_name}")
    elif llm_provider == "openai":
        chat_model_name = config.get("openai_chat_model", DEFAULT_OPENAI_CHAT_MODEL)
        embedding_model_name = config.get("openai_embedding_model", DEFAULT_OPENAI_EMBEDDING_MODEL)
        print(f"Using OpenAI provider with Chat Model: {chat_model_name}, Embedding Model: {embedding_model_name}")
    else:
        print(f"Error: Unsupported llm_provider '{llm_provider}' in config.json. Supported providers: 'google', 'openai'.")
        exit(1) # Exit if provider is invalid

    # --- Resume and job description paths ---
    app_resume_path = config.get("resume_path", DEFAULT_RESUME_PATH)
    job_description_path = config.get("job_description_path", DEFAULT_JOB_DESCRIPTION_PDF_PATH)
    
    # --- Run the App ---
    if os.path.exists(app_resume_path):
        try:
            interviewer = TechnicalInterviewerApp(
                # Pass resume and job description files path
                resume_path=app_resume_path,
                job_description_path=job_description_path,
                # Pass LLM provider and specific model names
                llm_provider=llm_provider,
                chat_model_name=chat_model_name,
                embedding_model_name=embedding_model_name
            )
            
            start_interview(interviewer)
        except Exception as e:
            print(f"\nAn error occurred during app execution: {e}")
            import traceback
            traceback.print_exc()
            print("Please ensure your API keys (GOOGLE_API_KEY / OPENAI_API_KEY) are set correctly in the .env file,")
            print("the resume path in config.json is valid, and necessary libraries (e.g., langchain-openai) are installed.")
    else:
        print(f"Error: Resume PDF not found at '{app_resume_path}'. Please create it or update the path in config.json.")
        