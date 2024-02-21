"""LM vs LM (LMvLM) hallucination detection method."""

from src.models.llm import BaseLlm


class Examiner:
    def __init__(self, claim, llm: BaseLlm):
        self.message_history = []
        self.claim = claim
        self.llm = llm

    def setup(self):
        prompt = (
            'Your goal is to try to verify the correctness of the following claim: \n<claim>\n{}\n</claim>\n, based on the background information you will gather. '
            'To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. '
            'Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. '
            'Please keep asking questions as long as you are yet to be sure regarding the true veracity of the claim. Please start with the first questions.'
        )
        message = {"role": "user", "content": prompt.format(self.claim)}
        self.message_history.append(message)
        response = self.generate_response()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        return response
    
    def check_follow_up_question(self, answer):
        prompt = '{}\n\n Do you have any follow-up questions? Please answer with Yes or No.'
        message = {"role": "user", "content": prompt.format(answer)}
        self.message_history.append(message)
        response = self.generate_response()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        return response
    
    def decision(self):
        prompt = (
            'Based on the interviewee\'s answers to your questions, what is your conclusion regarding the correctness of the claim? '
            'Do you think it is correct or incorrect?\nI think the claim is '
        )
        message = {"role": "user", "content": prompt}
        self.message_history.append(message)
        system_message = 'You are an expert at following guidelines for structuring text. Answer ONLY with "correct" or "incorrect".'
        messages = self.message_history
        messages.insert(-2, {"role": "system", "content": system_message})
        response = self.llm.get_response(messages, max_tokens=50)[-1]
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        return self.message_history
    
    def ask_continue(self):
        prompt = 'What are the follow-up questions?'
        message = {"role": "user", "content": prompt}
        self.message_history.append(message)
        response = self.generate_response()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        return response

    def generate_response(self):
        response = self.llm.get_response(self.message_history, max_tokens=256)[-1]
        return response


class Examinee:
    def __init__(self, claim, llm: BaseLlm):
        self.message_history = []
        self.message_history.append({"role": "assistant", "content": claim})
        self.llm = llm
        
    
    def answer_questions(self, question):
        prompt = 'Answer the following questions regarding your claim: \n<claim>\n{}\n</claim>'
        message = {"role": "user", "content": prompt.format(question)}
        self.message_history.append(message)
        response = self.generate_response()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        return response


    def generate_response(self):
        response = self.llm.get_response(self.message_history, max_tokens=256)[-1]
        return response


class LMvLM:
    def __init__(self, examinee_llm: BaseLlm, examiner_llm: BaseLlm):
        self.examinee_llm = examinee_llm
        self.examiner_llm = examiner_llm

    def get_hallucination_score(self, response: tuple, variants: list[str] = None) -> dict:

        resp_text = response[-1]
        examiner = Examiner(resp_text, self.examiner_llm)
        examinee = Examinee(resp_text, self.examinee_llm)
        question = examiner.setup()
        trigger = True
        count = 1
        while trigger:
            count += 1
            answer = examinee.answer_questions(question)
            flag = examiner.check_follow_up_question(answer)
            if 'No' in flag or count == 5:
                examiner_history = examiner.decision()
                trigger = False
            else:
                question = examiner.ask_continue()

        stop_reason = "Max turns reached" if count == 5 else "Examiner concluded"
                
        if 'incorrect' in examiner_history[-1]['content']:
            return {resp_text: {"score": 1, "stop_reason": stop_reason}}
        elif 'correct' in examiner_history[-1]['content']:
            return {resp_text: {"score": 0, "stop_reason": stop_reason}}
        else:
            return {resp_text: {"score": -1, "stop_reason": stop_reason}}