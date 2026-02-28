"""Add person_info_id to face_encodings for multi-embedding support

Revision ID: a1b2c3d4e5f6
Revises: 79ccbf3a85a1
Create Date: 2026-02-27 00:00:00.000000

Allows multiple face encodings to be linked to a single PersonInfo,
enabling recognition from multiple angles/distances.
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '79ccbf3a85a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add person_info_id FK column to face_encodings
    op.add_column('face_encodings',
        sa.Column('person_info_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'fk_face_encodings_person_info_id',
        'face_encodings', 'person_info',
        ['person_info_id'], ['id'],
        ondelete='SET NULL'
    )

    # Backfill: link existing encodings to their person via detected_faces -> person_info
    op.execute("""
        UPDATE face_encodings fe
        SET person_info_id = pi.id
        FROM detected_faces df
        JOIN person_info pi ON pi.face_id = df.id
        WHERE fe.face_id = df.id
    """)


def downgrade() -> None:
    op.drop_constraint('fk_face_encodings_person_info_id', 'face_encodings', type_='foreignkey')
    op.drop_column('face_encodings', 'person_info_id')
