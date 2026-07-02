import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

model = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="deepseek-v3-2-251201",  
    temperature=0.7,
    max_tokens=1024
)

# Redis 对话历史（Docker 容器间通过容器名通信）
history = RedisChatMessageHistory(
    session_id="my_session",
    redis_url="redis://redis:6379"
)

print("对话开始！输入 'quit' 退出，输入 'clear' 清空历史。")
print("-" * 50)

while True:
    user_input = input("\n你: ")
    
    if user_input.lower() == "quit":
        print("对话结束。")
        break
    
    if user_input.lower() == "clear":
        history.clear()
        print("[历史已清空]")
        continue
    
    # 添加用户消息到 Redis
    history.add_user_message(user_input)
    
    # 获取所有历史消息
    messages = history.messages
    
    # 调用模型
    response = model.invoke(messages)
    
    # 添加 AI 回复到 Redis
    history.add_ai_message(response.content)
    
    print(f"\nAI: {response.content}")
