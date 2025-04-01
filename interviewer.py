import os
import re
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
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- Pydantic Models for Structured Output Parsing ---

class WorkExperience(BaseModel):
    """Represents a single work experience entry."""
    job_title: str = Field(description="The job title held by the candidate.")
    company_name: str = Field(description="The name of the company.")
    start_date: Optional[str] = Field(description="The start date of the employment (e.g., YYYY-MM or Month YYYY).")
    end_date: Optional[str] = Field(description="The end date of the employment (e.g., YYYY-MM or Month YYYY), or 'Present'.")
    summary: str = Field(description="A brief summary of key responsibilities or achievements mentioned for this role.")

class ExperienceList(BaseModel):
    """Represents a list of work experiences extracted from the resume."""
    experiences: List[WorkExperience] = Field(description="A list of work experience objects.")

# --- Main Interviewer Application Class ---

class TechnicalInterviewerApp:
    """
    A LangChain application simulating a technical interviewer.
    """
    def __init__(self, resume_path: str, job_description: Optional[str] = None, model_name: str = "gemini-2.5-pro-exp-03-25"):
        """
        Initializes the interviewer app.

        Args:
            resume_path: Path to the candidate's resume PDF file.
            job_description: Optional text of the job description.
            model_name: The Google Generative AI model to use (e.g., "gemini-2.5-pro-exp-03-25").
        """
        print("Initializing Technical Interviewer App...")
        self.resume_path = resume_path
        self.job_description = job_description
        self.model_name = model_name
        self.interview_stage = "ICEBREAKER" # Initial stage
        self.coding_preferences = {} # To store language/difficulty

        # Initialize core components
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.7) #, convert_system_message_to_human=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Use the recommended embedding model
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load resume and extract structured experience
        self.resume_text, self.structured_experience = self._load_and_process_resume()
        if not self.resume_text:
            raise ValueError("Could not load or process the resume.")

        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # Define prompts and chains (will be created dynamically in methods)
        self.question_chain = None
        self.feedback_chain = None

        print("Initialization complete. Ready to start the interview.")

    def _load_and_process_resume(self) -> (Optional[str], Optional[List[Dict[str, Any]]]):
        """Loads the resume PDF, extracts text, and uses LLM to get structured experience."""
        print(f"Loading resume from: {self.resume_path}")
        try:
            loader = PyPDFLoader(self.resume_path)
            docs = loader.load()
            resume_text = "\n".join([doc.page_content for doc in docs])
            print("Resume loaded successfully.")

            print("Extracting structured experience using LLM...")
            # Define the parser and the prompt for experience extraction
            parser = JsonOutputParser(pydantic_object=ExperienceList)
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a resume analysis expert. Parse the following resume text and extract a list of professional work experiences. "
                    "For each experience, provide: 'job_title', 'company_name', 'start_date', 'end_date', and a 'summary' of key responsibilities/achievements. "
                    "Format the output as a JSON list adhering to the provided schema."
                    "\n{format_instructions}" # Instructions from the parser
                ),
                HumanMessagePromptTemplate.from_template("Resume Text:\n```{resume_text}```")
            ])

            # Create a chain specifically for parsing
            parsing_chain = prompt | self.llm | parser

            # Invoke the chain
            structured_data = parsing_chain.invoke({
                "resume_text": resume_text,
                "format_instructions": parser.get_format_instructions()
            })

            print("Structured experience extracted.")
            # Ensure it's a list before returning
            if isinstance(structured_data, dict) and 'experiences' in structured_data:
                 return resume_text, structured_data['experiences']
            else:
                print("Warning: LLM did not return the expected 'experiences' list structure.")
                return resume_text, [] # Return empty list if parsing failed

        except Exception as e:
            print(f"Error loading or processing resume: {e}")
            return None, None

    def _initialize_vectorstore(self) -> Optional[FAISS]:
        """Chunks documents, creates embeddings, and initializes FAISS vector store."""
        print("Initializing vector store...")
        try:
            # Combine resume text and JD text for chunking
            texts_to_embed = [self.resume_text]
            if self.job_description:
                texts_to_embed.append(self.job_description)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_splits = text_splitter.create_documents(texts_to_embed) # Pass list of texts

            print(f"Created {len(all_splits)} document chunks.")

            # Create FAISS vector store
            vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            print("FAISS vector store created successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None

    def _get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieves relevant context from the vector store."""
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
        print(f"\n--- Generating Question for Stage: {self.interview_stage} ---")

        # Base system message
        system_message = (
            "You are a friendly but professional technical interviewer. Your goal is to assess the candidate based on their resume, "
            "potentially a job description, and their answers. Ask relevant questions one at a time, manage the interview flow, "
            "and maintain an encouraging tone. After the candidate answers, you will evaluate their response (in a separate step)."
            f" The current interview stage is: {self.interview_stage}."
        )

        # Stage-specific instructions
        human_instructions = ""
        context = "" # Context from vector store

        if self.interview_stage == "ICEBREAKER":
            human_instructions = "Ask a simple, open-ended icebreaker question to start the conversation."

        elif self.interview_stage == "RECENT_EXPERIENCE":
            # Use structured experience primarily
            if self.structured_experience:
                 # Focus on the most recent role (or first in the list if order isn't guaranteed)
                recent_role = self.structured_experience[0]
                role_details = (
                    f"Title: {recent_role.get('job_title', 'N/A')}, "
                    f"Company: {recent_role.get('company_name', 'N/A')}, "
                    f"Dates: {recent_role.get('start_date', 'N/A')} - {recent_role.get('end_date', 'N/A')}. "
                    f"Summary: {recent_role.get('summary', 'N/A')}"
                )
                human_instructions = (
                    f"The candidate's most recent role based on extracted data is: {role_details}. "
                    "Based on this information and our conversation history, ask a specific question about a key project, challenge, or achievement mentioned in their summary for this role. "
                )
                # Optionally add context from JD if relevant
                if self.job_description:
                    jd_context = self._get_relevant_context(f"Skills related to {recent_role.get('job_title', '')} or {recent_role.get('summary', '')}", k=2)
                    human_instructions += f"\nConsider these potentially relevant points from the Job Description:\n{jd_context}"
            else:
                human_instructions = "Ask a general question about the candidate's most recent work experience or a significant project they worked on."
                context = self._get_relevant_context("recent work experience project achievement", k=2)

        elif self.interview_stage == "AI_BASICS":
             human_instructions = "The candidate wants to discuss AI. Ask a general question about their motivation, interest, or foundational understanding of AI/ML concepts."

        elif self.interview_stage == "AI_ADVANCED":
            context = self._get_relevant_context("AI machine learning deep learning neural networks LLMs computer vision technical design architecture", k=2)
            human_instructions = (
                "Based on our conversation history and potentially relevant context from their resume/JD, ask a more specific question about an AI/ML concept "
                "(e.g., supervised vs unsupervised, activation functions, transformers, CNNs, RNNs, model evaluation, MLOps, AI ethics, system design for an AI feature). "
                "Avoid focusing only on Generative AI unless the conversation leads there. Gradually increase complexity if appropriate."
                f"\nResume/JD Context:\n{context}"
            )

        elif self.interview_stage == "CODING_SETUP":
            human_instructions = "Ask the candidate what difficulty level (easy, medium, hard) and programming language (Python or Java) they prefer for the upcoming coding exercise."

        elif self.interview_stage == "CODING_CHALLENGE":
            lang = self.coding_preferences.get('language', 'Python')
            diff = self.coding_preferences.get('difficulty', 'medium')
            human_instructions = (
                f"Generate a {diff} level coding problem suitable for a technical interview, solvable in {lang}. "
                "The problem should ideally relate to common data structures, algorithms, or basic logic. Provide only the problem description clearly."
            )

        else: # Fallback or END stage
            return "Thank you for your time today. This concludes the interview."

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"), # Memory
            HumanMessagePromptTemplate.from_template(human_instructions)
        ])

        # Create and run the chain
        self.question_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False # Set to True for debugging prompts
        )
        response = self.question_chain.predict(input="") # Pass empty input as instructions are in the template
        return response

    def _evaluate_answer(self, question: str, answer: str) -> str:
        """Evaluates the candidate's answer and provides feedback."""
        print("--- Evaluating Answer ---")

        system_message = (
            "You are an AI assistant evaluating a candidate's answer during a technical interview. "
            "Your goal is to provide constructive feedback."
        )
        human_instructions = (
            f"Here was the question asked:\n'''{question}'''\n\n"
            f"Here is the candidate's answer:\n'''{answer}'''\n\n"
            "Based on the question and answer, please provide specific, constructive feedback. "
            "Mention what was good about the answer (e.g., clarity, depth, specific examples, correctness, relevance). "
            "Also, point out areas for improvement or what might have been lacking (e.g., missing detail, technical inaccuracy, need for better structure, edge cases missed for code). "
            "Be encouraging but objective. If evaluating code, assess correctness, efficiency (mention Big O if applicable), style, and edge cases."
            "\n\nFeedback:"
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_instructions)
        ])

        # Create and run the chain (no memory needed for direct evaluation)
        self.feedback_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )
        feedback = self.feedback_chain.predict(question=question, answer=answer) # Pass vars directly
        return feedback

    def _transition_state(self, user_input: str):
        """Transitions the interview stage based on user input or logic."""
        lower_input = user_input.lower()

        # --- Explicit Transitions based on Keywords ---
        if "talk about ai" in lower_input or "discuss ai" in lower_input or "ask about ai" in lower_input:
            if self.interview_stage not in ["AI_BASICS", "AI_ADVANCED"]:
                print("Transitioning state to AI_BASICS")
                self.interview_stage = "AI_BASICS"
                return # Transitioned
        elif "coding challenge" in lower_input or "coding exercise" in lower_input or "let's code" in lower_input:
             if self.interview_stage != "CODING_SETUP":
                print("Transitioning state to CODING_SETUP")
                self.interview_stage = "CODING_SETUP"
                return # Transitioned
        elif "stop" in lower_input or "end interview" in lower_input or "finish" in lower_input:
             print("Transitioning state to END")
             self.interview_stage = "END"
             return # Transitioned

        # --- Implicit Transitions based on Stage Logic ---
        if self.interview_stage == "ICEBREAKER":
            # Automatically move to experience after the first question
            print("Transitioning state to RECENT_EXPERIENCE")
            self.interview_stage = "RECENT_EXPERIENCE"
        elif self.interview_stage == "RECENT_EXPERIENCE":
            # Simple logic: move to AI basics after a few turns (e.g., 2 questions)
            # A more robust approach could involve LLM deciding or checking coverage
            history = self.memory.chat_memory.messages
            experience_turns = sum(1 for i in range(len(history) - 1, 0, -2) if history[i-1].content.startswith("The candidate's most recent role")) # Count questions related to experience
            if experience_turns >= 1: # Move after 1-2 experience questions
                 # Ask user if they want to move on (optional refinement)
                 print("Transitioning state to AI_BASICS (heuristic)")
                 self.interview_stage = "AI_BASICS"
        elif self.interview_stage == "CODING_SETUP":
             # Try to parse difficulty and language
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
                 # Stay in CODING_SETUP stage

        elif self.interview_stage == "CODING_CHALLENGE":
            # After providing feedback on a solution, ask if they want another or to end
            if "another" in lower_input or "next problem" in lower_input:
                print("Transitioning state back to CODING_SETUP")
                self.interview_stage = "CODING_SETUP" # Ask for preferences again
            elif "conclude" in lower_input or "that's all" in lower_input:
                 print("Transitioning state to END")
                 self.interview_stage = "END"
            # Otherwise, stay in CODING_CHALLENGE (implies user is providing solution)


    def start_interview(self):
        """Starts the interactive interview loop."""
        print("\n--- Interview Started ---")
        print("Type 'stop', 'end interview', or 'finish' at any time to end.")
        print("You can also say 'talk about AI' or 'coding challenge' to switch topics.")

        while self.interview_stage != "END":
            # 1. Generate Question
            question = self._generate_question()
            if self.interview_stage == "END": # Check if generation triggered end
                 print(f"\nInterviewer: {question}")
                 break
            print(f"\nInterviewer: {question}")

            # 2. Get User Answer
            user_answer = input("You: ")

             # 2a. Check for immediate stop command
            if user_answer.lower() in ["stop", "end interview", "finish"]:
                self.interview_stage = "END"
                print("\nInterviewer: Understood. Thank you for your time today!")
                break

            # 3. Evaluate Answer & Provide Feedback (Skip for setup stage)
            if self.interview_stage != "CODING_SETUP":
                feedback = self._evaluate_answer(question, user_answer)
                print(f"\nInterviewer Feedback: {feedback}")
            else:
                 # Don't evaluate the preference answer, just proceed
                 pass

            # 4. Update Memory (Done implicitly by LLMChain with memory)
            # Note: The feedback itself isn't added to memory unless we explicitly do:
            # self.memory.save_context({"input": f"Feedback on previous answer"}, {"output": feedback})

            # 5. Transition State
            self._transition_state(user_answer) # Transition based on the user's *last* response

        print("\n--- Interview Ended ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    RESUME_PDF_PATH = "Linkedin José Arrarte.pdf" # <--- CHANGE THIS TO YOUR RESUME PDF PATH
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
    if not os.path.exists(RESUME_PDF_PATH):
         try:
             from reportlab.pdfgen import canvas
             from reportlab.lib.pagesizes import letter
             print(f"Creating dummy PDF: {RESUME_PDF_PATH}")
             c = canvas.Canvas(RESUME_PDF_PATH, pagesize=letter)
             textobject = c.beginText(40, 750)
             textobject.textLine("Alex Chen")
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
    if os.path.exists(RESUME_PDF_PATH):
        try:
            interviewer = TechnicalInterviewerApp(
                resume_path=RESUME_PDF_PATH,
                job_description=JOB_DESCRIPTION_TEXT
            )
            interviewer.start_interview()
        except Exception as e:
            print(f"\nAn error occurred during app execution: {e}")
            print("Please ensure your GOOGLE_API_KEY is set correctly in the .env file and the PDF path is valid.")
    else:
        print(f"Error: Resume PDF not found at '{RESUME_PDF_PATH}'. Please create it or update the path.")

