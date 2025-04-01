import os
import re
import json # Import the json library
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# LangChain components for document loading, splitting, embeddings, vector stores
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain components for LLMs, chat models, memory, prompts, and parsers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
# Use Pydantic v2 imports directly
from pydantic import BaseModel, Field

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- Pydantic Models for Structured Output Parsing (Updated) ---

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

# --- Main Interviewer Application Class ---

class TechnicalInterviewerApp:
    """
    A LangChain application simulating a technical interviewer.
    """
    # Default model name remains as a fallback if config loading fails or key is missing
    def __init__(self, resume_path: str, job_description: Optional[str] = None, model_name: str = "gemini-1.5-pro-latest"):
        """
        Initializes the interviewer app.

        Args:
            resume_path: Path to the candidate's resume PDF file.
            job_description: Optional text of the job description.
            model_name: The Google Generative AI model to use.
        """
        print("Initializing Technical Interviewer App...")
        print(f"Using model: {model_name}") # Log the model being used
        self.resume_path = resume_path # Store the path passed during instantiation
        self.job_description = job_description
        self.model_name = model_name
        self.interview_stage = "ICEBREAKER" # Initial stage
        self.coding_preferences = {} # To store language/difficulty
        self.candidate_name = None # Initialize candidate name

        # Initialize core components
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load resume and extract structured data (name + experience)
        # Uses self.resume_path which was set from the argument
        self.resume_text, self.candidate_name, self.structured_experience = self._load_and_process_resume()
        if not self.resume_text:
            # Raise or handle error appropriately if resume loading is critical
             print(f"Warning: Could not load or process the resume at {self.resume_path}. Some features might be limited.")
             # Decide if execution should continue; here we allow it but print a warning.
             # raise ValueError(f"Could not load or process the resume at {self.resume_path}.")
        elif self.candidate_name:
            print(f"Successfully extracted candidate name: {self.candidate_name}")
        else:
            print("Could not extract candidate name from resume.")


        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # Define prompts and chains (will be created dynamically in methods)
        self.question_chain = None
        self.feedback_chain = None

        print("Initialization complete. Ready to start the interview.")

    def _load_and_process_resume(self) -> (Optional[str], Optional[str], Optional[List[Dict[str, Any]]]):
        """Loads resume PDF, extracts text, and uses LLM to get structured name and experience."""
        # Check if the path exists before trying to load
        if not os.path.exists(self.resume_path):
             print(f"Error: Resume file not found at {self.resume_path}")
             return None, None, []

        print(f"Loading resume from: {self.resume_path}")
        resume_text = None
        candidate_name = None
        experiences = []
        try:
            loader = PyPDFLoader(self.resume_path)
            docs = loader.load()
            resume_text = "\n".join([doc.page_content for doc in docs])
            if not resume_text:
                 print("Warning: No text extracted from PDF.")
                 return None, None, []
            print("Resume loaded successfully.")

            print("Extracting structured name and experience using LLM...")
            parser = JsonOutputParser(pydantic_object=ResumeData)
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a resume analysis expert. Parse the following resume text. "
                    "Extract the candidate's full name and a list of professional work experiences. "
                    "For the name, identify the most likely full name of the candidate. "
                    "For each experience, provide: 'job_title', 'company_name', 'start_date', 'end_date', and a 'summary' of key responsibilities/achievements. "
                    "Format the output as a JSON object adhering to the provided schema, including 'candidate_name' and 'experiences' keys."
                    "\n{format_instructions}"
                ),
                HumanMessagePromptTemplate.from_template("Resume Text:\n```{resume_text}```")
            ])
            parsing_chain = prompt | self.llm | parser
            structured_data = parsing_chain.invoke({
                "resume_text": resume_text,
                "format_instructions": parser.get_format_instructions()
            })
            print("Structured data extracted.")

            if isinstance(structured_data, dict):
                candidate_name = structured_data.get('candidate_name')
                experiences = structured_data.get('experiences', [])
                if not experiences: print("Warning: No experiences extracted.")
                if not isinstance(experiences, list):
                    print("Warning: 'experiences' field is not a list.")
                    experiences = []
            else:
                 print("Warning: LLM did not return the expected dictionary structure.")

            return resume_text, candidate_name, experiences

        except Exception as e:
            print(f"Error loading or processing resume: {e}")
            return resume_text, candidate_name, experiences # Return potentially partial data

    def _initialize_vectorstore(self) -> Optional[FAISS]:
        """Chunks documents, creates embeddings, and initializes FAISS vector store."""
        if not self.resume_text:
             print("Skipping vector store initialization: No resume text.")
             return None
        # ... (rest of the method is unchanged) ...
        print("Initializing vector store...")
        try:
            texts_to_embed = [self.resume_text]
            if self.job_description:
                texts_to_embed.append(self.job_description)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_splits = text_splitter.create_documents(texts_to_embed)
            print(f"Created {len(all_splits)} document chunks.")
            vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            print("FAISS vector store created successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None

    def _get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieves relevant context from the vector store."""
        # ... (method is unchanged) ...
        if not self.vectorstore:
            return "No vector store available."
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = "\n---\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return "Error retrieving context."

    def _generate_question(self) -> str:
        """Generates the next interview question based on the current stage."""
        # ... (method is largely unchanged, uses self.candidate_name, self.structured_experience etc.) ...
        print(f"\n--- Generating Question for Stage: {self.interview_stage} ---")
        system_message = (
            "You are a friendly but professional technical interviewer. Your goal is to assess the candidate based on their resume, "
            "potentially a job description, and their answers. Ask relevant questions one at a time, manage the interview flow, "
            "and maintain an encouraging tone. After the candidate answers, you will evaluate their response (in a separate step)."
            f" The current interview stage is: {self.interview_stage}."
        )
        if self.candidate_name:
             system_message += f" The candidate's name is {self.candidate_name}."
        human_instructions = ""
        context = ""
        if self.interview_stage == "ICEBREAKER":
            human_instructions = "Ask a simple, open-ended icebreaker question to start the conversation."
            if self.candidate_name and not self.memory.chat_memory.messages:
                 human_instructions = f"Start by greeting {self.candidate_name} by name, then ask a simple, open-ended icebreaker question."
        elif self.interview_stage == "RECENT_EXPERIENCE":
            if self.structured_experience:
                if not self.structured_experience:
                    human_instructions = "Ask a general question about the candidate's work experience or a significant project they worked on, as specific details couldn't be extracted."
                else:
                    recent_role = self.structured_experience[0]
                    role_details = (f"Title: {recent_role.get('job_title', 'N/A')}, Company: {recent_role.get('company_name', 'N/A')}, Dates: {recent_role.get('start_date', 'N/A')} - {recent_role.get('end_date', 'N/A')}. Summary: {recent_role.get('summary', 'N/A')}")
                    human_instructions = (f"The candidate's most recent role based on extracted data is: {role_details}. Based on this information and our conversation history, ask a specific question about a key project, challenge, or achievement mentioned in their summary for this role. ")
                    if self.job_description:
                        jd_context = self._get_relevant_context(f"Skills related to {recent_role.get('job_title', '')} or {recent_role.get('summary', '')}", k=2)
                        if jd_context and "Error" not in jd_context and "No vector store" not in jd_context:
                            human_instructions += f"\nConsider these potentially relevant points from the Job Description:\n{jd_context}"
            else:
                human_instructions = "Ask a general question about the candidate's most recent work experience or a significant project they worked on, as specific details couldn't be extracted."
                context = self._get_relevant_context("recent work experience project achievement", k=2)
                if context and "Error" not in context and "No vector store" not in context:
                     human_instructions += f"\nPotentially relevant context from resume:\n{context}"
        elif self.interview_stage == "AI_BASICS":
             human_instructions = "The candidate wants to discuss AI. Ask a general question about their motivation, interest, or foundational understanding of AI/ML concepts."
        elif self.interview_stage == "AI_ADVANCED":
            context = self._get_relevant_context("AI machine learning deep learning neural networks LLMs computer vision technical design architecture", k=2)
            human_instructions = ("Based on our conversation history and potentially relevant context from their resume/JD, ask a more specific question about an AI/ML concept (e.g., supervised vs unsupervised, activation functions, transformers, CNNs, RNNs, model evaluation, MLOps, AI ethics, system design for an AI feature). Avoid focusing only on Generative AI unless the conversation leads there. Gradually increase complexity if appropriate.")
            if context and "Error" not in context and "No vector store" not in context:
                 human_instructions += f"\nResume/JD Context:\n{context}"
        elif self.interview_stage == "CODING_SETUP":
            human_instructions = "Ask the candidate what difficulty level (easy, medium, hard) and programming language (Python or Java) they prefer for the upcoming coding exercise."
        elif self.interview_stage == "CODING_CHALLENGE":
            lang = self.coding_preferences.get('language', 'Python')
            diff = self.coding_preferences.get('difficulty', 'medium')
            human_instructions = (f"Generate a {diff} level coding problem suitable for a technical interview, solvable in {lang}. The problem should ideally relate to common data structures, algorithms, or basic logic. Provide only the problem description clearly.")
        else:
            return "Thank you for your time today. This concludes the interview."
        prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_message), MessagesPlaceholder(variable_name="chat_history"), HumanMessagePromptTemplate.from_template(human_instructions)])
        if self.question_chain is None or self.question_chain.prompt.messages != prompt.messages:
             self.question_chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory, verbose=False)
        response = self.question_chain.predict(input="")
        return response

    def _evaluate_answer(self, question: str, answer: str) -> str:
        """Evaluates the candidate's answer and provides feedback."""
        # ... (method is unchanged) ...
        print("--- Evaluating Answer ---")
        system_message = ("You are an AI assistant evaluating a candidate's answer during a technical interview. Your goal is to provide constructive feedback.")
        human_instructions = (f"Here was the question asked:\n'''{question}'''\n\nHere is the candidate's answer:\n'''{answer}'''\n\nBased on the question and answer, please provide specific, constructive feedback. Mention what was good about the answer (e.g., clarity, depth, specific examples, correctness, relevance). Also, point out areas for improvement or what might have been lacking (e.g., missing detail, technical inaccuracy, need for better structure, edge cases missed for code). Be encouraging but objective. If evaluating code, assess correctness, efficiency (mention Big O if applicable), style, and edge cases.\n\nFeedback:")
        prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_message), HumanMessagePromptTemplate.from_template(human_instructions)])
        if self.feedback_chain is None or self.feedback_chain.prompt.messages != prompt.messages:
            self.feedback_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
        feedback = self.feedback_chain.predict(question=question, answer=answer)
        return feedback

    def _transition_state(self, user_input: str):
        """Transitions the interview stage based on user input or logic."""
        # ... (method is unchanged) ...
        lower_input = user_input.lower()
        current_stage = self.interview_stage
        if "talk about ai" in lower_input or "discuss ai" in lower_input or "ask about ai" in lower_input:
            if self.interview_stage not in ["AI_BASICS", "AI_ADVANCED"]:
                print("Transitioning state to AI_BASICS")
                self.interview_stage = "AI_BASICS"
                return
        elif "coding challenge" in lower_input or "coding exercise" in lower_input or "let's code" in lower_input:
             if self.interview_stage != "CODING_SETUP":
                print("Transitioning state to CODING_SETUP")
                self.interview_stage = "CODING_SETUP"
                return
        elif "stop" in lower_input or "end interview" in lower_input or "finish" in lower_input:
             print("Transitioning state to END")
             self.interview_stage = "END"
             return
        if current_stage == "ICEBREAKER":
            print("Transitioning state to RECENT_EXPERIENCE")
            self.interview_stage = "RECENT_EXPERIENCE"
        elif current_stage == "RECENT_EXPERIENCE":
            history = self.memory.chat_memory.messages
            experience_questions_count = 0
            for i in range(0, len(history), 2):
                 msg_content = history[i].content.lower()
                 if "recent role" in msg_content or "work experience" in msg_content or ("project" in msg_content and "achievement" in msg_content):
                     if i > 0 or "icebreaker" not in msg_content :
                        experience_questions_count += 1
            if experience_questions_count >= 1:
                 print(f"Transitioning state to AI_BASICS (heuristic after {experience_questions_count} experience questions)")
                 self.interview_stage = "AI_BASICS"
        elif current_stage == "CODING_SETUP":
             difficulty_match = re.search(r'\b(easy|medium|hard)\b', lower_input)
             language_match = re.search(r'\b(python|java)\b', lower_input, re.IGNORECASE)
             if difficulty_match and language_match:
                 self.coding_preferences['difficulty'] = difficulty_match.group(1)
                 self.coding_preferences['language'] = language_match.group(1).capitalize()
                 print(f"Coding preferences set: {self.coding_preferences}")
                 print("Transitioning state to CODING_CHALLENGE")
                 self.interview_stage = "CODING_CHALLENGE"
             else:
                 print("Could not detect both difficulty (easy/medium/hard) and language (Python/Java). Please specify.")
        elif current_stage == "CODING_CHALLENGE":
            if "another" in lower_input or "next problem" in lower_input:
                print("Transitioning state back to CODING_SETUP")
                self.interview_stage = "CODING_SETUP"
            elif "conclude" in lower_input or "that's all" in lower_input or "end coding" in lower_input:
                 print("Transitioning state to END")
                 self.interview_stage = "END"

    def start_interview(self):
        """Starts the interactive interview loop."""
        # ... (method is unchanged) ...
        print("\n--- Interview Started ---")
        greeting = "Hello!"
        if self.candidate_name:
            greeting = f"Hello {self.candidate_name}!"
        print(f"Interviewer: {greeting} Welcome to the technical interview simulation.")
        print("Type 'stop', 'end interview', or 'finish' at any time to end.")
        print("You can also say 'talk about AI' or 'coding challenge' to switch topics.")
        while self.interview_stage != "END":
            question = self._generate_question()
            if self.interview_stage == "END":
                 print(f"\nInterviewer: {question}")
                 break
            print(f"\nInterviewer: {question}")
            is_coding_challenge_question = (self.interview_stage == "CODING_CHALLENGE" and ("generate a" in question.lower() or "coding problem" in question.lower()) and ("difficulty" in question.lower() or self.coding_preferences))
            user_answer = input("You: ")
            if user_answer.lower() in ["stop", "end interview", "finish"]:
                self.interview_stage = "END"
                print("\nInterviewer: Understood. Thank you for your time today!")
                break
            if self.interview_stage != "CODING_SETUP" and not is_coding_challenge_question:
                feedback = self._evaluate_answer(question, user_answer)
                print(f"\nInterviewer Feedback: {feedback}")
            elif is_coding_challenge_question:
                 print("\nInterviewer: Okay, thank you for providing your solution. I will evaluate it.")
                 pass
            else:
                 pass
            self._transition_state(user_answer)
        print("\n--- Interview Ended ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration Loading ---
    CONFIG_FILE_PATH = "config.json"
    DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25" # Fallback default
    DEFAULT_RESUME_PATH = "placeholder_resume.pdf" # Fallback default

    config = {}
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {CONFIG_FILE_PATH}")
    except FileNotFoundError:
        print(f"Warning: Configuration file '{CONFIG_FILE_PATH}' not found. Using default values.")
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{CONFIG_FILE_PATH}'. Using default values.")

    # Get parameters from config or use defaults
    app_model_name = config.get("model_name", DEFAULT_MODEL)
    app_resume_path = config.get("resume_path", DEFAULT_RESUME_PATH)

    # --- Optional Job Description ---
    # Keep JD loading separate for now, could be added to config later if needed
    JOB_DESCRIPTION_TEXT = """
    Software Engineer - AI/ML

    We are looking for a talented Software Engineer with a passion for Artificial Intelligence
    and Machine Learning. You will work on developing and deploying ML models, building data pipelines,
    and contributing to our AI platform.

    Requirements:
    - BS/MS in Computer Science or related field.
    - 2+ years of experience in software development (Python preferred).
    - Experience with ML frameworks (TensorFlow, PyTorch, scikit-learn).
    - Understanding of ML concepts (supervised/unsupervised learning, model evaluation).
    - Experience with cloud platforms (GCP, AWS, Azure) is a plus.
    - Strong problem-solving skills.
    """ # <--- Optional: Replace with actual JD text or set to None

    # --- Create a Dummy PDF for testing if needed ---
    # Use the resume path loaded from config
    if not os.path.exists(app_resume_path):
         try:
             from reportlab.pdfgen import canvas
             from reportlab.lib.pagesizes import letter
             print(f"Creating dummy PDF: {app_resume_path}")
             c = canvas.Canvas(app_resume_path, pagesize=letter)
             textobject = c.beginText(40, 750)
             c.setFont("Helvetica-Bold", 16)
             textobject.textLine("Alex Chen")
             c.setFont("Helvetica", 12)
             textobject.moveCursor(0, 5)
             textobject.textLine("Software Engineer | AI Enthusiast")
             textobject.textLine("alex.chen@email.com | 555-1234 | linkedin.com/in/alexchen")
             textobject.moveCursor(0, 20)
             textobject.textLine("Summary")
             textobject.textLine("Driven software engineer with 3 years of experience...")
             textobject.moveCursor(0, 20)
             textobject.textLine("Experience")
             textobject.textLine("Software Engineer, Tech Solutions Inc. (Jan 2022 - Present)")
             textobject.textLine("- Developed a recommendation engine using Python and scikit-learn, improving user engagement by 15%.")
             textobject.textLine("- Built data processing pipelines using Apache Spark.")
             textobject.moveCursor(0, 10)
             textobject.textLine("Junior Developer, Web Widgets Co. (Jun 2020 - Dec 2021)")
             textobject.textLine("- Contributed to front-end development using React.")
             textobject.moveCursor(0, 20)
             textobject.textLine("Education")
             textobject.textLine("B.S. Computer Science, State University (2020)")
             textobject.moveCursor(0, 20)
             textobject.textLine("Skills")
             textobject.textLine("Python, Java, SQL, scikit-learn, TensorFlow, Docker, Git, AWS")
             c.drawText(textobject)
             c.save()
         except ImportError:
             print("Please install reportlab (`pip install reportlab`) to create a dummy PDF.")
         except Exception as e:
             print(f"Error creating dummy PDF: {e}")


    # --- Run the App ---
    # Use the parameters loaded from the config file
    if os.path.exists(app_resume_path): # Check existence using loaded path
        try:
            interviewer = TechnicalInterviewerApp(
                resume_path=app_resume_path,       # Pass loaded resume path
                job_description=JOB_DESCRIPTION_TEXT,
                model_name=app_model_name          # Pass loaded model name
            )
            interviewer.start_interview()
        except Exception as e:
            print(f"\nAn error occurred during app execution: {e}")
            print("Please ensure your GOOGLE_API_KEY is set correctly in the .env file and the resume path in config.json is valid.")
    else:
        # Use loaded path in error message
        print(f"Error: Resume PDF not found at '{app_resume_path}'. Please create it or update the path in config.json.")

