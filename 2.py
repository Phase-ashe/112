import os
import re
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

model = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="deepseek-v3-2-251201",  
    temperature=0,
    max_tokens=1024
)

# 系统提示词
SYSTEM_PROMPT = """你是一个有帮助的AI助手。你拥有多种工具能力：
- 当用户提到之前聊过的内容、早期对话、或你不确定是否记得的话题时，调用 search_history 工具从 Redis 历史中检索。
- 当用户询问天气时，调用 get_weather 工具获取实时天气信息。
- 当用户询问日期、时间、星期时，调用 get_current_datetime 工具获取准确时间。
请根据用户意图自主判断是否需要调用工具。"""

# 上下文压缩配置
MAX_TOKEN_WINDOW = 100  # 最大 token 窗口
COMPRESS_RATIO = 0.6     # 超过窗口后，保留最近 60% 的消息，压缩其余部分


def estimate_tokens(text: str) -> int:
    """估算 token 数：中文约 1.5 字符/token，英文约 4 字符/token"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def estimate_messages_tokens(messages) -> int:
    """估算消息列表的总 token 数"""
    return sum(estimate_tokens(m.content) for m in messages)


def compress_context(messages: list, max_tokens: int) -> list:
    """
    仿照 Claude Code 的上下文压缩：
    当总 token 超过限制时，将旧消息用模型压缩为摘要，保留最近的消息。
    """
    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages

    print(f"\n[上下文压缩] 当前约 {total} tokens，超过上限 {max_tokens}，正在压缩...")

    # 分离系统提示词和普通消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    normal_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # 分割：要压缩的旧消息 + 要保留的近期消息
    split_idx = max(1, int(len(normal_msgs) * (1 - COMPRESS_RATIO)))
    old_msgs = normal_msgs[:split_idx]
    recent_msgs = normal_msgs[split_idx:]

    if not old_msgs:
        return messages

    # 构造压缩摘要的对话内容
    compress_text = ""
    for msg in old_msgs:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        compress_text += f"{role}: {msg.content}\n"

    # 调用模型生成摘要
    summary_prompt = [
        SystemMessage(content="你是一个对话摘要助手。请将以下对话内容压缩为简洁的摘要，保留关键信息和上下文，去除冗余内容。直接输出摘要，不要加任何前缀。"),
        HumanMessage(content=f"请压缩以下对话：\n\n{compress_text}")
    ]
    summary_response = model.invoke(summary_prompt)
    summary_text = summary_response.content

    # 用摘要替换旧消息
    compressed = system_msgs + [
        SystemMessage(content=f"[历史对话摘要] {summary_text}")
    ] + recent_msgs

    new_total = estimate_messages_tokens(compressed)
    print(f"[上下文压缩] 压缩完成，约 {new_total} tokens。")
    print("\n压缩后的消息列表：")
    print("-" * 40)
    for i, msg in enumerate(compressed, 1):
        if isinstance(msg, SystemMessage):
            tag = "系统" if not msg.content.startswith("[历史对话摘要]") else "摘要"
        elif isinstance(msg, HumanMessage):
            tag = "用户"
        else:
            tag = "AI"
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  [{i}] [{tag}] {preview}")
    print("-" * 40)
    return compressed


def query_redis_history(keyword: str, max_results: int = 10) -> str:
    """
    从 Redis 完整历史中查询对话记录。
    可用于查找很久之前的对话，或被上下文压缩压缩掉的旧消息。

    Args:
        keyword: 搜索关键词，会在用户和AI的消息内容中模糊匹配
        max_results: 最大返回条数，默认 10

    Returns:
        匹配到的历史对话记录文本
    """
    all_messages = history.messages
    results = []
    for msg in all_messages:
        if isinstance(msg, SystemMessage):
            continue
        if keyword.lower() in msg.content.lower():
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            results.append(f"{role}: {msg.content}")
        if len(results) >= max_results:
            break

    if results:
        return f"找到 {len(results)} 条相关历史记录：\n" + "\n".join(results)
    else:
        return f"未找到与 '{keyword}' 相关的历史记录。"


# 注册为 LangChain tool，供 Agent 调用
@tool
def search_history(keyword: str) -> str:
    """从 Redis 完整历史中搜索对话记录。当用户提到之前聊过的内容、早期对话、或需要回忆被压缩的旧消息时，调用此工具检索。"""
    return query_redis_history(keyword)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。当用户询问某个城市的天气时调用此工具。"""
    # 模拟天气数据（实际可接入 OpenWeatherMap 等 API）
    weather_data = {
        "北京": {"temp": 28, "condition": "晴", "humidity": 45, "wind": "东北风 3级"},
        "上海": {"temp": 30, "condition": "多云", "humidity": 65, "wind": "东南风 2级"},
        "广州": {"temp": 33, "condition": "雷阵雨", "humidity": 80, "wind": "南风 2级"},
        "深圳": {"temp": 32, "condition": "阴", "humidity": 75, "wind": "西南风 3级"},
        "杭州": {"temp": 29, "condition": "小雨", "humidity": 70, "wind": "东风 2级"},
        "成都": {"temp": 26, "condition": "阴转多云", "humidity": 60, "wind": "北风 1级"},
    }
    # 支持随机生成未预设城市的数据
    if city in weather_data:
        w = weather_data[city]
    else:
        w = {
            "temp": random.randint(18, 35),
            "condition": random.choice(["晴", "多云", "阴", "小雨", "雷阵雨"]),
            "humidity": random.randint(30, 90),
            "wind": random.choice(["东风", "南风", "西风", "北风"]) + f" {random.randint(1,5)}级"
        }
    return json.dumps({
        "city": city,
        "temperature": f"{w['temp']}°C",
        "condition": w["condition"],
        "humidity": f"{w['humidity']}%",
        "wind": w["wind"],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }, ensure_ascii=False)


@tool
def get_current_datetime() -> str:
    """获取当前的日期、时间和星期。当用户询问今天日期、现在几点、星期几时调用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": weekdays[now.weekday()],
        "timestamp": now.isoformat()
    }, ensure_ascii=False)


# 创建 ReAct Agent：自动判断是否需要调用工具
agent = create_react_agent(
    model=model,
    tools=[search_history, get_weather, get_current_datetime],
    prompt=SYSTEM_PROMPT
)


# Redis 对话历史（Docker 容器间通过容器名通信）
history = RedisChatMessageHistory(
    session_id="my_session",
    redis_url="redis://redis:6379"
)

# 初始化时将系统提示词写入 Redis（仅当历史为空时）
if not history.messages:
    history.add_message(SystemMessage(content=SYSTEM_PROMPT))

print("对话开始！输入 'quit' 退出，输入 'clear' 清空历史。")
print("输入 'history' 查看所有历史对话，输入 'search <关键词>' 搜索历史消息。")
print("-" * 50)

while True:
    user_input = input("\n你: ")
    
    if user_input.lower() == "quit":
        print("对话结束。")
        break
    
    if user_input.lower() == "clear":
        history.clear()
        history.add_message(SystemMessage(content=SYSTEM_PROMPT))
        print("[历史已清空]")
        continue
    
    # 查看所有历史对话
    if user_input.lower() == "history":
        all_messages = history.messages
        print("\n" + "=" * 40)
        print("历史对话记录：")
        print("=" * 40)
        idx = 1
        for msg in all_messages:
            if isinstance(msg, SystemMessage):
                continue
            role = "你" if isinstance(msg, HumanMessage) else "AI"
            print(f"  [{idx}] {role}: {msg.content}")
            idx += 1
        if idx == 1:
            print("  （暂无对话记录）")
        print("=" * 40)
        continue
    
    # 搜索历史消息
    if user_input.lower().startswith("search "):
        keyword = user_input[7:].strip()
        if not keyword:
            print("[请输入要搜索的关键词，例如: search 你好]")
            continue
        all_messages = history.messages
        results = []
        for msg in all_messages:
            if isinstance(msg, SystemMessage):
                continue
            if keyword.lower() in msg.content.lower():
                role = "你" if isinstance(msg, HumanMessage) else "AI"
                results.append((role, msg.content))
        print(f"\n搜索 '{keyword}' 的结果（共 {len(results)} 条）：")
        print("-" * 40)
        if results:
            for i, (role, content) in enumerate(results, 1):
                print(f"  [{i}] {role}: {content}")
        else:
            print("  （未找到匹配结果）")
        print("-" * 40)
        continue
    
    # 添加用户消息到 Redis
    history.add_user_message(user_input)
    
    # 获取所有历史消息（系统提示词已存储在 Redis 中）
    messages = history.messages
    
    # 显示当前 token 估算（调试用）
    current_tokens = estimate_messages_tokens(messages)
    print(f"\n[调试] 当前上下文约 {current_tokens} / {MAX_TOKEN_WINDOW} tokens")
    
    # 上下文压缩：超过 token 窗口时自动压缩旧消息
    messages = compress_context(messages, MAX_TOKEN_WINDOW)
    
    # Agent 自动判断是否需要调用工具，流式输出
    print("\nAI: ", end="", flush=True)
    full_response = ""
    
    for event in agent.stream({"messages": messages}, stream_mode="updates"):
        # 处理工具调用事件
        if "tools" in event:
            for msg in event["tools"]["messages"]:
                print(f"\n[工具] {msg.content[:120]}...")
        # 处理 Agent 输出事件
        if "agent" in event:
            for msg in event["agent"]["messages"]:
                if msg.content and not msg.tool_calls:
                    print(msg.content, end="", flush=True)
                    full_response += msg.content
    
    print()  # 换行
    
    # 添加 AI 回复到 Redis
    if full_response:
        history.add_ai_message(full_response)
