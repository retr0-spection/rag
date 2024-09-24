"""modify messages table, added hidden column

Revision ID: 4ea8099936b7
Revises: 382d73e71c93
Create Date: 2024-09-22 09:57:02.579260

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4ea8099936b7'
down_revision: Union[str, None] = '382d73e71c93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('hidden', sa.Boolean))


def downgrade() -> None:
    pass
