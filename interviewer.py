import os
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# LangChain components for document loading, splitting, vector stores
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Provider Specific Imports ---
# Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# --- End Provider Specific Imports ---

# LangChain components for memory, prompts, and parsers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- Pydantic Models ---
class WorkExperience(BaseModel):
    """Represents a single work experience entry."""
    job_title: str = Field(description="The job title held by the candidate.")
    company_name: str = Field(description="The name of the company.")
    start_date: Optional[str] = Field(description="The start date of the employment (e.g., YYYY-MM or Month YYYY).")
    end_date: Optional[str] = Field(description="The end date of the employment (e.g., YYYY-MM or Month YYYY), or 'Present'.")
    summary: str = Field(description="A brief summary of key responsibilities or achievements mentioned for this role.")

class ResumeData(BaseModel):
    """Represents structured data extracted from the resume."""
    candidate_name: Optional[str] = Field(default=None, description="The full name of the candidate identified in the resume.")
    experiences: List[WorkExperience] = Field(description="A list of work experience objects.")

# class JobDescriptionData(BaseModel):
#     """Represents structured data extracted from the job description."""
#     job_title: Optional[str] = Field(default=None, description="The job title mentioned in the job description.")
#     company_name: Optional[str] = Field(default=None, description="The name of the company.")
#     requirements: List[str] = Field(description="A list of requirements or qualifications mentioned in the job description.")
#     responsibilities: List[str] = Field(description="A list of responsibilities mentioned in the job description.")

# --- Main Interviewer Application Class ---

class TechnicalInterviewerApp:
    """
    A LangChain application simulating a technical interviewer.
    """
    def __init__(
        self,
        resume_path: str,
        job_description_path: str,
        llm_provider: str,            
        chat_model_name: str,         
        embedding_model_name: str,    
        temperature: float = 0.7
        
    ):
        """
        Initializes the interviewer app.

        Args:
            resume_path: Path to the candidate's resume PDF file.
            llm_provider: The LLM provider ('google' or 'openai').
            chat_model_name: The specific chat model name for the provider.
            embedding_model_name: The specific embedding model name for the provider.
            job_description: Optional text of the job description.
            temperature: The temperature setting for the LLM.
        """
        print(f"Initializing Technical Interviewer App using provider: {llm_provider}...")
        print(f"Chat Model: {chat_model_name}, Embedding Model: {embedding_model_name}")

        self.resume_path = resume_path
        self.job_description_path = job_description_path
        self.interview_stage = "ICEBREAKER"
        self.coding_preferences = {}
        self.candidate_name = None

        # --- Initialize LLM and Embeddings based on provider ---
        self.llm = None
        self.embeddings = None

        if llm_provider == "google":
            # Ensure GOOGLE_API_KEY is set in .env
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            self.llm = ChatGoogleGenerativeAI(model=chat_model_name, temperature=temperature) 
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
            print("Initialized Google LLM and Embeddings.")
        elif llm_provider == "openai":
            # Ensure OPENAI_API_KEY is set in .env
            if not os.getenv("OPENAI_API_KEY"):
                 raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
            try:
                self.llm = ChatOpenAI(model=chat_model_name, temperature=temperature)
                self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
                print("Initialized OpenAI LLM and Embeddings.")
            except ImportError:
                 print("\nERROR: langchain-openai package not found.")
                 print("Please install it: pip install langchain-openai\n")
                 raise
            except Exception as e:
                 print(f"\nERROR initializing OpenAI components: {e}")
                 print("Ensure your OPENAI_API_KEY is valid and the model names are correct.")
                 raise
        else:
            # Should have been caught in __main__, but good to double-check
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")
        
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load job description 
        self.job_description_text = self._load_job_description()
        if not self.job_description_text:
             print(f"Warning: Could not load or process the job description at {self.job_description_path}. Some features might be limited.")
            
        # Load resume and extract structured data
        self.resume_text, self.candidate_name, self.structured_experience = self._load_and_process_resume()
        if not self.resume_text:
             print(f"Warning: Could not load or process the resume at {self.resume_path}. Some features might be limited.")
        elif self.candidate_name:
            print(f"Successfully extracted candidate name: {self.candidate_name}")
        else:
            print("Could not extract candidate name from resume.")

        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # --- Define Chains ---
        question_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_message}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_instructions}")
        ])
        base_question_chain = question_prompt | self.llm

        # Wrap the base chain with history management
        self.question_chain_with_history = RunnableWithMessageHistory(
            base_question_chain,
            lambda session_id: self.memory.chat_memory, # Returns the BaseChatMessageHistory object
            # REMOVE input_messages_key: Let keys pass through directly
            # input_messages_key="input_dict",
            history_messages_key="chat_history", # Matches MessagesPlaceholder
        )

        # Feedback chain definition
        feedback_system_message = (
            "You are an AI assistant evaluating a candidate's answer during a technical interview. "
            "Your goal is to provide constructive feedback."
        )
        feedback_human_template = (
            "Here was the question asked:\n'''{question}'''\n\n"
            "Here is the candidate's answer:\n'''{answer}'''\n\n"
            "Based on the question and answer, please provide specific, constructive feedback. "
            "Mention what was good about the answer (e.g., clarity, depth, specific examples, correctness, relevance). "
            "Also, point out areas for improvement or what might have been lacking (e.g., missing detail, technical inaccuracy, need for better structure, edge cases missed for code). "
            "Be encouraging but objective. If evaluating code, assess correctness, efficiency (mention Big O if applicable), style, and edge cases."
            "\n\nFeedback:"
        )
        feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(feedback_system_message),
            HumanMessagePromptTemplate.from_template(feedback_human_template)
        ])
        self.feedback_chain = feedback_prompt | self.llm

        print("Initialization complete. Ready to start the interview.")

    # --- Helper methods for dynamic prompt content ---
    def _get_system_message(self) -> str:
        """Builds the system message dynamically."""
        system_message = (
            "You are a friendly but professional technical interviewer. Your goal is to assess the candidate based on their resume, "
            "potentially a job description, and their answers. Ask relevant questions one at a time, manage the interview flow, "
            "and maintain an encouraging tone. After the candidate answers, you will evaluate their response (in a separate step)."
            f" The current interview stage is: {self.interview_stage}."
        )
        if self.candidate_name:
             system_message += f" The candidate's name is {self.candidate_name}."
             
        system_message += (
            " Review the recent conversation history provided. "
            "Do NOT ask the exact same question that you asked in the immediately preceding turn. "
            "Ask the next logical question based on the current interview stage and the flow of the conversation. If appropriate, you can ask a follow-up question related to the candidate's last answer."
        )

        return system_message

    def _get_human_instructions_for_stage(self) -> str:
        """Builds the human instructions based on the interview stage."""
        human_instructions = ""

        if self.interview_stage == "ICEBREAKER":
            human_instructions = "Ask a simple, open-ended icebreaker question to start the conversation."
            # Check history correctly using load_memory_variables
            # Use .get("chat_history", []) to handle the case where history might be None initially
            if self.candidate_name and not self.memory.load_memory_variables({}).get("chat_history", []):
                 human_instructions = f"Start by greeting {self.candidate_name} by name, then ask a simple, open-ended icebreaker question."

        elif self.interview_stage == "RECENT_EXPERIENCE":
            if self.structured_experience: # Check if we have structured experience data
                try:
                    # Get the most recent experience (assuming the first item is the most recent)
                    recent_role = self.structured_experience[0]

                    # Safely get details, providing fallbacks if keys are missing
                    job_title = recent_role.get("job_title", "their most recent role")
                    company_name = recent_role.get("company_name", "their last company")
                    summary = recent_role.get("summary", "") # Get the summary too for context

                    # Construct specific instructions using the extracted details
                    human_instructions = (
                        f"The candidate's resume lists their most recent position as '{job_title}' at '{company_name}'. "
                    )
                    if summary:
                        # Add summary context if available
                        human_instructions += f"The summary mentioned: \"{summary}\". "
                    human_instructions += (
                        f"Ask a specific question focusing on their key responsibilities, a significant project, or a challenge they faced in that role at {company_name}. "
                        "Avoid overly generic questions like 'Tell me about your last job'."
                    )

                except (IndexError, TypeError, AttributeError) as e:
                    # Fallback if structured_experience is not a list, empty, or item isn't a dict/object
                    print(f"Warning: Could not access recent role details from structured_experience: {e}. Falling back to general question.")
                    human_instructions = "Ask a general question about the candidate's most recent work experience or a significant project they worked on."
            else:
                # Fallback if no structured experience was extracted at all
                human_instructions = "Ask a general question about the candidate's most recent work experience or a significant project they worked on."

        elif self.interview_stage == "AI_BASICS":
             human_instructions = "Ask a fundamental question about AI or Machine Learning concepts relevant to a Software Engineer role (e.g., difference between supervised/unsupervised learning, overfitting, activation functions)."
        elif self.interview_stage == "AI_ADVANCED":
             human_instructions = "Ask a more advanced or practical AI/ML question (e.g., explain a specific algorithm like Transformers or CNNs, discuss MLOps, model deployment strategies, or handling imbalanced data)."
        elif self.interview_stage == "CODING_SETUP":
             human_instructions = "Ask the candidate their preferred programming language (e.g., Python, Java) and desired difficulty level (easy, medium, hard) for the upcoming coding challenge."
        elif self.interview_stage == "CODING_CHALLENGE":
             if self.coding_preferences:
                 human_instructions = f"Generate a {self.coding_preferences.get('difficulty', 'medium')} coding problem suitable for a technical interview, solvable in {self.coding_preferences.get('language', 'Python')}. Present the problem clearly."
             else:
                 human_instructions = "Generate a medium difficulty coding problem suitable for a technical interview (e.g., array manipulation, string processing, basic data structures). Ask them to explain their approach first." # Fallback if preferences not set
        else: # END stage
            human_instructions = "The interview is concluding. Provide a polite closing remark, thank the candidate for their time, and briefly mention next steps if applicable (though you are an AI, so keep it general)."

        return human_instructions

    # --- Core Methods ---
    def _load_and_process_resume(self) -> (Optional[str], Optional[str], Optional[List[Dict[str, Any]]]):
        """Loads resume PDF, extracts text, and uses LLM to get structured name and experience."""
        
        if not os.path.exists(self.resume_path): 
            print(f"Error: Resume file not found at {self.resume_path}"); 
            return None, None, []
        
        print(f"Loading resume from: {self.resume_path}"); 
        resume_text = None; 
        candidate_name = None; 
        experiences = []
        
        try:
            loader = PyPDFLoader(self.resume_path); 
            docs = loader.load(); 
            resume_text = "\n".join([doc.page_content for doc in docs])
            if not resume_text: 
                print("Warning: No text extracted from PDF."); 
                return None, None, []
            print("Resume loaded successfully."); 
            print("Extracting structured name and experience using LLM...")
            parser = JsonOutputParser(pydantic_object=ResumeData)
            prompt = ChatPromptTemplate.from_messages([ SystemMessagePromptTemplate.from_template("... Parse resume ...\n{format_instructions}"), HumanMessagePromptTemplate.from_template("Resume Text:\n```{resume_text}```")])
            parsing_chain = prompt | self.llm | parser
            structured_data = parsing_chain.invoke({"resume_text": resume_text, "format_instructions": parser.get_format_instructions()})
            print("Structured data extracted.")
            if isinstance(structured_data, dict): 
                candidate_name = structured_data.get('candidate_name'); 
                experiences = structured_data.get('experiences', []) 
                # ... error checks ...
            else: 
                print("Warning: LLM did not return expected structure.")
                
            return resume_text, candidate_name, experiences
        except Exception as e: print(f"Error loading or processing resume: {e}"); return resume_text, candidate_name, experiences

    def _load_job_description(self) -> Optional[str]:
    #def _load_job_description(self) -> (Optional[str], Optional[str], Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]):
        """Loads job description PDF, extracts text, and uses LLM to get structured name and experience."""
        
        if not os.path.exists(self.job_description_path): 
            print(f"Error: Job description file not found at {self.job_description_path}"); 
            return None, None, []
        
        print(f"Loading job description from: {self.job_description_path}"); 
        job_description_text = None;
        
        try:
            loader = PyPDFLoader(self.job_description_path); 
            docs = loader.load(); 
            job_description_text = "\n".join([doc.page_content for doc in docs])
            if not job_description_text: print("Warning: No text extracted from PDF."); return None, None, []
            print("Job description loaded successfully."); 
            
            # parser = JsonOutputParser(pydantic_object=JobDescriptionData)
            # prompt = ChatPromptTemplate.from_messages([ SystemMessagePromptTemplate.from_template("... Parse job description ...\n{format_instructions}"), HumanMessagePromptTemplate.from_template("Job description Text:\n```{job_description_text}```")])
            # parsing_chain = prompt | self.llm | parser
            # structured_data = parsing_chain.invoke({"job_description_text": job_description_text, "format_instructions": parser.get_format_instructions()})
            # print("Job description Structured data extracted.")
            
            # if isinstance(structured_data, dict): 
            #     job_title = structured_data.get('job_title')
            #     company_name = structured_data.get('company_name')
            #     requirements = structured_data.get('requirements')
            #     responsibilities = structured_data.get('responsibilities')
            #     # ... error checks ...
            # else: 
            #     print("Warning: LLM did not return expected structure.")
                
            return job_description_text
        except Exception as e: print(f"Error loading or processing job description: {e}"); 
        
        return job_description_text #, job_title, company_name, requirements, responsibilities


    def _initialize_vectorstore(self) -> Optional[FAISS]:
        """Chunks documents, creates embeddings, and initializes FAISS vector store."""
        
        if not self.resume_text: print("Skipping vector store initialization: No resume text."); return None
        print("Initializing vector store...");
        try:
            texts_to_embed = [self.resume_text];
            if self.job_description: 
                texts_to_embed.append(self.job_description_text)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150); all_splits = text_splitter.create_documents(texts_to_embed)
            print(f"Created {len(all_splits)} document chunks."); vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            print("FAISS vector store created successfully."); return vectorstore
        except Exception as e: print(f"Error initializing vector store: {e}"); return None

    def _get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieves relevant context from the vector store."""
        
        if not self.vectorstore: return "No vector store available."
        try: docs = self.vectorstore.similarity_search(query, k=k); context = "\n---\n".join([doc.page_content for doc in docs]); return context
        except Exception as e: print(f"Error during similarity search: {e}"); return "Error retrieving context."

    def _generate_question(self) -> str:
        """Generates the next interview question using the history-aware chain."""
        print(f"\n--- Generating Question for Stage: {self.interview_stage} ---")
        if self.interview_stage == "END":
             return "Thank you for your time today. This concludes the interview."

        system_message = self._get_system_message()
        human_instructions = self._get_human_instructions_for_stage()

        # Prepare the input dictionary with keys directly expected by the prompt template
        input_vars = {
            "system_message": system_message,
            "human_instructions": human_instructions
            # 'chat_history' will be added automatically by RunnableWithMessageHistory
        }

        print(f"Invoking question chain with input vars: {list(input_vars.keys())}") # Debug print

        response = self.question_chain_with_history.invoke(
            # Pass the variables directly, not nested
            input_vars,
            config={"configurable": {"session_id": "interview_session"}}
        )
        question_content = response.content
        print(f"Generated question: {question_content}") # Debug print
        return question_content
    

    def _evaluate_answer(self, question: str, answer: str) -> str:
        """Evaluates the candidate's answer and provides feedback using LCEL."""
        
        print("--- Evaluating Answer ---")
        response = self.feedback_chain.invoke({"question": question, "answer": answer})
        feedback_content = response.content
        return feedback_content

    def _transition_state(self, user_input: str):
        """Transitions the interview stage based on user input or logic."""
        lower_input = user_input.lower()
        current_stage = self.interview_stage # Keep track of the stage *before* potential changes

        # --- Keyword-based transitions (check these first) ---
        if "talk about ai" in lower_input or "discuss ai" in lower_input or "ask about ai" in lower_input:
            if self.interview_stage not in ["AI_BASICS", "AI_ADVANCED"]:
                print("Transitioning state to AI_BASICS based on user request")
                self.interview_stage = "AI_BASICS"
                return # Exit after transition
        elif "coding challenge" in lower_input or "coding exercise" in lower_input or "let's code" in lower_input:
             if self.interview_stage != "CODING_SETUP":
                 print("Transitioning state to CODING_SETUP based on user request")
                 self.interview_stage = "CODING_SETUP"
                 return # Exit after transition
        elif "stop" in lower_input or "end interview" in lower_input or "finish" in lower_input:
             print("Transitioning state to END based on user request")
             self.interview_stage = "END"
             return # Exit after transition

        # --- Automatic stage progression ---
        # Check the stage *before* keyword transitions might have changed it
        if current_stage == "ICEBREAKER":
            print("Transitioning state from ICEBREAKER to RECENT_EXPERIENCE")
            self.interview_stage = "RECENT_EXPERIENCE"
        elif current_stage == "RECENT_EXPERIENCE":
            # --- SIMPLIFIED TRANSITION ---
            # Automatically move to the next stage after one question in RECENT_EXPERIENCE
            print(f"Transitioning state from RECENT_EXPERIENCE to AI_BASICS")
            self.interview_stage = "AI_BASICS"
            # Remove the complex counting logic
        elif current_stage == "AI_BASICS":
             # Decide when to move from AI_BASICS to AI_ADVANCED or other stage
             # For now, let's assume one question here too, then maybe coding setup?
             print(f"Transitioning state from AI_BASICS to AI_ADVANCED") # Or CODING_SETUP? Adjust flow as needed.
             self.interview_stage = "AI_ADVANCED" # Or CODING_SETUP
        elif current_stage == "AI_ADVANCED":
             # After advanced AI, maybe move to coding setup?
             print(f"Transitioning state from AI_ADVANCED to CODING_SETUP")
             self.interview_stage = "CODING_SETUP"
        elif current_stage == "CODING_SETUP":
             # This transition *requires* user input, so check preferences
             difficulty_match = re.search(r'\b(easy|medium|hard)\b', lower_input)
             language_match = re.search(r'\b(python|java)\b', lower_input, re.IGNORECASE)
             if difficulty_match and language_match:
                 self.coding_preferences['difficulty'] = difficulty_match.group(1)
                 self.coding_preferences['language'] = language_match.group(1).capitalize()
                 print(f"Coding preferences set: {self.coding_preferences}")
                 print("Transitioning state from CODING_SETUP to CODING_CHALLENGE")
                 self.interview_stage = "CODING_CHALLENGE"
             else:
                 # Stay in CODING_SETUP if preferences aren't set
                 print("Could not detect both difficulty (easy/medium/hard) and language (Python/Java). Please specify.")
                 # Ensure the stage doesn't change accidentally if only one part was matched
                 self.interview_stage = "CODING_SETUP"
        elif current_stage == "CODING_CHALLENGE":
            # This transition depends on user wanting another problem or ending
            if "another" in lower_input or "next problem" in lower_input:
                print("Transitioning state back to CODING_SETUP for next challenge")
                self.interview_stage = "CODING_SETUP"
            elif "conclude" in lower_input or "that's all" in lower_input or "end coding" in lower_input:
                print("Transitioning state from CODING_CHALLENGE to END")
                self.interview_stage = "END"
            # Otherwise, stay in CODING_CHALLENGE (e.g., if user provides solution/asks question)

        # Ensure END state persists
        if self.interview_stage == "END":
             print("Interview state is END.")

    
    def start_interview(self):
        """Starts the interactive interview loop."""
        
        print("\n--- Interview Started ---"); greeting = "Hello!";
        if self.candidate_name: greeting = f"Hello {self.candidate_name}!"
        print(f"Interviewer: {greeting} Welcome to the technical interview simulation.")
        print("Type 'stop', 'end interview', or 'finish' at any time to end.")
        print("You can also say 'talk about AI' or 'coding challenge' to switch topics.")
        while self.interview_stage != "END":
            question = self._generate_question()
            if self.interview_stage == "END": print(f"\nInterviewer: {question}"); break
            print(f"\nInterviewer: {question}")
            is_coding_challenge_question = (self.interview_stage == "CODING_CHALLENGE" and ("generate a" in question.lower() or "coding problem" in question.lower()) and ("difficulty" in question.lower() or self.coding_preferences))
            user_answer = input("You: ")
            if user_answer.lower() in ["stop", "end interview", "finish"]: self.interview_stage = "END"; print("\nInterviewer: Understood. Thank you for your time today!"); break
            if self.interview_stage != "CODING_SETUP" and not is_coding_challenge_question:
                feedback = self._evaluate_answer(question, user_answer)
                print(f"\nInterviewer Feedback: {feedback}")
            elif is_coding_challenge_question: print("\nInterviewer: Okay, thank you for providing your solution. I will evaluate it.")
            self._transition_state(user_answer)
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
            interviewer.start_interview()
        except Exception as e:
            print(f"\nAn error occurred during app execution: {e}")
            import traceback
            traceback.print_exc()
            print("Please ensure your API keys (GOOGLE_API_KEY / OPENAI_API_KEY) are set correctly in the .env file,")
            print("the resume path in config.json is valid, and necessary libraries (e.g., langchain-openai) are installed.")
    else:
        print(f"Error: Resume PDF not found at '{app_resume_path}'. Please create it or update the path in config.json.")
        