"""Your migration message

Revision ID: 382d73e71c93
Revises: 5c73ac5849ec
Create Date: 2024-09-15 00:55:02.430746

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '382d73e71c93'
down_revision: Union[str, None] = '5c73ac5849ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
