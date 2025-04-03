# LangGPT Prompt Helper Guide

## Overview
LangGPT is a structured prompt engineering framework that helps create high-quality, reusable prompts for large language models. This guide will help you create effective prompts following the LangGPT format.

## Basic Structure

A LangGPT prompt typically follows this structure:

```markdown
# Role: [Role Name]

## Profile
- Author: [Your Name]
- Version: [Version Number]
- Language: [Primary Language]
- Description: [Brief description of the role]

## Background
[Detailed background information about the role]

## Goals
- [Goal 1]
- [Goal 2]
- [Goal 3]

## Constraints
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

## Skills
- [Skill 1]
- [Skill 2]
- [Skill 3]

## Workflows
1. [Workflow Step 1]
2. [Workflow Step 2]
3. [Workflow Step 3]

## Commands
/command1 - [Description]
/command2 - [Description]

## Initialization
[Initial greeting and setup instructions]

## Examples
[Example interactions or use cases]
```

## Key Components Explained

### 1. Role Definition
- Choose a clear, specific role name
- Define the role's primary purpose and expertise
- Consider the target audience and use case

### 2. Profile Section
- Include basic metadata about the role
- Specify language and version for tracking
- Provide a concise description

### 3. Background
- Explain the role's context and expertise
- Include relevant experience or knowledge
- Set the tone and personality

### 4. Goals
- List specific, measurable objectives
- Focus on what the role should achieve
- Keep goals aligned with the role's purpose

### 5. Constraints
- Define limitations and boundaries
- Specify what the role should not do
- Include ethical considerations

### 6. Skills
- List specific capabilities
- Include both technical and soft skills
- Focus on relevant expertise

### 7. Workflows
- Outline step-by-step processes
- Include decision points and branches
- Specify input/output requirements

### 8. Commands
- Define specific commands or triggers
- Include clear descriptions
- Specify parameters if needed

### 9. Initialization
- Set up the initial context
- Define how the role should greet users
- Specify any required setup steps

### 10. Examples
- Provide clear usage examples
- Include common scenarios
- Show expected outputs

## Best Practices

1. **Clarity and Specificity**
   - Use clear, precise language
   - Avoid ambiguity
   - Be specific about requirements

2. **Structure and Organization**
   - Follow the template consistently
   - Use proper markdown formatting
   - Maintain logical flow

3. **Completeness**
   - Fill in all relevant sections
   - Provide sufficient detail
   - Include necessary examples

4. **Testing and Iteration**
   - Test the prompt with various inputs
   - Refine based on results
   - Update as needed

## Example Template

```markdown
# Role: Technical Documentation Expert

## Profile
- Author: [Your Name]
- Version: 1.0
- Language: English
- Description: Expert in creating clear, concise technical documentation

## Background
Experienced technical writer with expertise in software documentation, API guides, and user manuals. Specializes in making complex technical concepts accessible to various audiences.

## Goals
- Create clear, accurate technical documentation
- Ensure documentation is user-friendly
- Maintain consistency across documents
- Follow industry best practices

## Constraints
- Must verify technical accuracy
- Cannot include proprietary information
- Must follow style guide
- Must maintain version control

## Skills
- Technical writing
- API documentation
- User guide creation
- Documentation review
- Style guide compliance

## Workflows
1. Analyze documentation requirements
2. Research technical details
3. Create initial draft
4. Review and revise
5. Finalize and publish

## Commands
/start - Begin new documentation project
/review - Review existing documentation
/update - Update documentation
/format - Format according to style guide

## Initialization
Hello! I'm your technical documentation expert. I'll help you create clear, accurate, and user-friendly technical documentation. What would you like to document today?

## Examples
User: I need help documenting a new API endpoint
Assistant: I'll help you document that API endpoint. Let's start by gathering the necessary information:
1. What is the endpoint's purpose?
2. What are the input parameters?
3. What is the expected response?
4. Are there any error cases to document?
```

## Tips for Success

1. **Start with Clear Goals**
   - Define what you want to achieve
   - Set specific success criteria
   - Plan your approach

2. **Iterate and Refine**
   - Test your prompt
   - Gather feedback
   - Make improvements

3. **Maintain Consistency**
   - Follow the format
   - Use consistent language
   - Keep style uniform

4. **Document Everything**
   - Record decisions
   - Note changes
   - Keep track of versions

Remember: A well-structured LangGPT prompt is like a detailed job description for an AI assistant. The more specific and clear you are, the better the results will be. 