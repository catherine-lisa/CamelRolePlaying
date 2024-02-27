# 角色扮演聊天系统

本项目是论文的langchain复现: 《CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society”》，

一个角色扮演聊天系统，旨在促进“Python程序员”（助手）和“股票交易员”（用户）之间的协作。系统的主要目标是开发一个股票市场的交易机器人。

## 功能特点

- **角色定义**：系统清晰地定义了助手和用户的角色。助手是Python程序员，用户是股票交易员。
- **任务细化**：系统从一个通用任务开始，并将其细化为更具体的任务。这是通过使用预定义模板和OpenAI聊天模型来完成的。
- **互动对话**：系统支持助手和用户之间的互动对话。助手根据用户的指示提供解决方案，用户根据他们的需求提供指示。
- **任务完成**：系统旨在通过这种互动对话来完成指定的任务。

## 实现方式

- **CAMELAgent**：系统的核心是 `CAMELAgent` 类，它管理助手和用户之间的对话。
  - **初始化**：代理使用系统消息和OpenAI聊天模型进行初始化。
  - **消息管理**：代理跟踪助手和用户之间交换的消息。
  - **步骤函数**：代理的步骤函数更新消息，与OpenAI模型交互，并返回输出消息。
- **预定义提示**：系统使用预定义的提示来指导助手和用户之间的对话。这些提示确保角色不会翻转，对话保持专注于任务完成。

## 使用方法

要使用系统，只需运行脚本。系统将开始细化任务，然后继续进行助手和用户之间的互动对话。对话将持续进行，直到任务完成。

## 系统要求

- Python 3.x
- OpenAI API密钥（需存储在`.env`文件中）
- 所需的Python包（如导入语句中所提到的）

### 运行示例

```python
# 定义自己想要的角色
assistant_role_name = "Python Programmer"
user_role_name = "Stock Trader"
task = "Develop a trading bot for the stock market"

# 开始对话    
chat_turn_limit, n = 10, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):{user_msg.content}")

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):{assistant_msg.content}")

    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
```
初步运行后果保存在output.md文件中（不同对话结果不同，仅供参考）。


### 注意事项
- 项目需要有效的API密钥来访问OpenAI模型。
- 项目可能需要进一步的测试和优化，以确保稳定性和性能。
- 项目需要添加自己的.env环境，配置对应的OPEN_API_KEY