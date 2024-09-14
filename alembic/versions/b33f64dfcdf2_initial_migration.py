"""Initial migration

Revision ID: b33f64dfcdf2
Revises: 
Create Date: 2024-09-14 14:15:50.991965

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = 'b33f64dfcdf2'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('projects')
    op.drop_index('ix_knowledge_bases_id', table_name='knowledge_bases')
    op.drop_table('knowledge_bases')
    op.drop_table('agent_knowledge_base_association')
    op.drop_index('ix_files_filename', table_name='files')
    op.drop_table('files')
    op.drop_index('ix_api_keys_id', table_name='api_keys')
    op.drop_index('ix_api_keys_key', table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_table('agents')
    op.drop_index('ix_sessions_id', table_name='sessions')
    op.drop_table('sessions')
    op.drop_index('ix_messages_id', table_name='messages')
    op.drop_table('messages')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('users',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('email', sa.VARCHAR(), nullable=True),
    sa.Column('hashed_password', sa.VARCHAR(), nullable=True),
    sa.Column('is_active', sa.BOOLEAN(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=1)
    op.create_table('messages',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('session_id', sa.INTEGER(), nullable=False),
    sa.Column('sender', sa.VARCHAR(), nullable=False),
    sa.Column('content', sa.TEXT(), nullable=False),
    sa.Column('timestamp', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_messages_id', 'messages', ['id'], unique=False)
    op.create_table('sessions',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('agent_id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.VARCHAR(), nullable=False),
    sa.Column('start_time', sa.DATETIME(), nullable=True),
    sa.Column('end_time', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sessions_id', 'sessions', ['id'], unique=False)
    op.create_table('agents',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(length=100), nullable=False),
    sa.Column('agent_type', sa.VARCHAR(length=50), nullable=False),
    sa.Column('user_id', sa.VARCHAR(), nullable=True),
    sa.Column('configuration', sqlite.JSON(), nullable=True),
    sa.Column('description', sa.TEXT(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('api_keys',
    sa.Column('id', sa.VARCHAR(), nullable=False),
    sa.Column('key', sa.VARCHAR(), nullable=True),
    sa.Column('user_id', sa.INTEGER(), nullable=True),
    sa.Column('permissions', sa.VARCHAR(), nullable=True),
    sa.Column('rate_limit', sa.INTEGER(), nullable=True),
    sa.Column('expiry_date', sa.DATETIME(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_keys_key', 'api_keys', ['key'], unique=1)
    op.create_index('ix_api_keys_id', 'api_keys', ['id'], unique=False)
    op.create_table('files',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('filename', sa.VARCHAR(), nullable=True),
    sa.Column('file_path', sa.VARCHAR(), nullable=True),
    sa.Column('content_type', sa.VARCHAR(), nullable=True),
    sa.Column('size', sa.INTEGER(), nullable=True),
    sa.Column('upload_date', sa.DATETIME(), nullable=True),
    sa.Column('last_modified', sa.DATETIME(), nullable=True),
    sa.Column('is_public', sa.BOOLEAN(), nullable=True),
    sa.Column('description', sa.VARCHAR(), nullable=True),
    sa.Column('owner_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_files_filename', 'files', ['filename'], unique=False)
    op.create_table('agent_knowledge_base_association',
    sa.Column('agent_id', sa.INTEGER(), nullable=False),
    sa.Column('knowledge_base_id', sa.INTEGER(), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['knowledge_base_id'], ['knowledge_bases.id'], ),
    sa.PrimaryKeyConstraint('agent_id', 'knowledge_base_id')
    )
    op.create_table('knowledge_bases',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(), nullable=False),
    sa.Column('description', sa.TEXT(), nullable=True),
    sa.Column('data', sa.TEXT(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_knowledge_bases_id', 'knowledge_bases', ['id'], unique=False)
    op.create_table('projects',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('last_modified', sa.DATETIME(), nullable=True),
    sa.Column('description', sa.VARCHAR(), nullable=True),
    sa.Column('owner_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###
