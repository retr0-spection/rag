from app.utils.auth import auth_dependency
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.models import Agent, User  # Import your SQLAlchemy models

router = APIRouter(tags=["agents"])

# Models for incoming data
#
#
class AgentConfig(BaseModel):
    prompt: str
    llm: str

    class Config:
        orm_mode = True


class AgentCreate(BaseModel):
    name: str
    agent_type: str
    configuration: AgentConfig

    class Config:
        orm_mode = True

class AgentUpdate(BaseModel):
    name: str
    agent_type: str
    configuration: AgentConfig

class AgentResponse(BaseModel):
    id: int
    name: str
    agent_type: str
    configuration: AgentConfig

    class Config:
        orm_mode = True  # This allows SQLAlchemy models to be converted to Pydantic models

# Route to create an agent
@router.post('/agents', response_model=AgentResponse)
def create_agent(agent: AgentCreate, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    # Check if the associated user exists
    user = db.query(User).filter_by(id=user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_agent = Agent(name=agent.name, user_id=user.id, configuration=agent.configuration.json())
    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)

    return new_agent

# Route to retrieve an agent by ID
@router.get('/agents/{agent_id}', response_model=AgentResponse)
def get_agent(agent_id: int, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    agent = db.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return agent

# Route to update an agent
@router.put('/agents/{agent_id}', response_model=AgentResponse)
def update_agent(agent_id: int, agent: AgentUpdate, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    existing_agent = db.query(Agent).filter_by(id=agent_id).first()
    if not existing_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update agent information
    existing_agent.name = agent.name
    existing_agent.user_id = agent.user_id
    db.commit()

    return existing_agent

# Route to delete an agent
@router.delete('/agents/{agent_id}', response_model=dict)
def delete_agent(agent_id: int, db: Session = Depends(get_db), user = Depends(auth_dependency)):
    agent = db.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    db.delete(agent)
    db.commit()

    return {"message": "Agent deleted successfully"}

# Route to get all agents
@router.get('/agents', response_model=list[AgentResponse])
def get_all_agents(user = Depends(auth_dependency), db: Session = Depends(get_db)):
    agents = db.query(Agent).filter_by(user_id=user.id)
    return agents
