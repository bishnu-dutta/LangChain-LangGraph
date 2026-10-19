from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-langgraph!")
    information = """
    Paul Carlyle (born 19 July 1967) is a Northern Irish former footballer and currently a manager.
    Carlyle, nicknamed 'Storky', played for Derry City in the League of Ireland and was an important member of their historic treble-winning squad in the 1988–89 season. In June 1989 after a trial at Liverpool he represented the League of Ireland in a quadrangular international tournament in Trinidad and Tobago.[1]
    He scored what is remembered as a great goal on a momentous European Cup occasion in the Brandywell against Benfica when he drilled a shot past the keeper into the roof of the net.[2]
    He signed for Shamrock Rovers in August 1987 and made his debut on the 23rd at Longford. In total he made twenty total appearances including two in the 1987–88 European Cup.
    He made his League of Ireland debut for Derry at Cobh Ramblers on 19 October 1986 after moving from Coleraine.
    Trialed at Manchester United in February 1990.
    His older brother Hilary played for Finn Harps, Dundalk (where he scored in the European Cup) and in the North American Soccer League.
    Paul (Storky) is a lifelong Liverpool supporter.
    Paul is a former manager of Institute, having taken up the role in 2011 after the resignation of John Gregg and leaving the club in December 2012.[3]
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["Information"],
        template="""
        You are a helpful assistant. 
        Please summarize the following information: {Information}
        """
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    chain = summary_prompt_template | llm | StrOutputParser()
    response = chain.invoke(input={"Information": information})
    print(response)


if __name__ == "__main__":
    main()
