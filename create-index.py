import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from inferedge_moss import DocumentInfo, MossClient

load_dotenv()


async def upload_documents():
    """Upload the receptionist knowledge base to a Moss index.

    Keep each document to a single fact — hours, one policy, one service —
    rather than a large block of text. Small, single-topic chunks retrieve
    more precisely than a whole paragraph or policy doc dumped as one chunk.
    """
    logger.debug("Starting the document upload process...")

    client = MossClient(
        project_id=os.getenv("MOSS_PROJECT_ID"), project_key=os.getenv("MOSS_PROJECT_KEY")
    )

    # TODO: replace every value below with your real business info.
    documents = [
        DocumentInfo(
            id="hours-1",
            text="We're open Monday to Saturday, 9am to 7pm. We're closed on Sundays and major public holidays.",
            metadata={"category": "hours", "topic": "business_hours"},
        ),
        DocumentInfo(
            id="hours-2",
            text="We do accept walk-ins, but booking an appointment in advance means shorter wait times.",
            metadata={"category": "hours", "topic": "walk_ins"},
        ),
        DocumentInfo(
            id="services-1",
            text="Our core services include consultations, routine checkups, and follow-up visits. Ask about a specific service if you don't see it listed.",
            metadata={"category": "services", "topic": "overview"},
        ),
        DocumentInfo(
            id="pricing-1",
            text="A standard consultation costs $75. Follow-up visits within 30 days are $40. Prices may vary depending on the service requested.",
            metadata={"category": "pricing", "topic": "consultation_fees"},
        ),
        DocumentInfo(
            id="policy-cancellation",
            text="Cancellations or reschedules require at least 24 hours notice. Cancellations made with less notice may incur a $25 fee.",
            metadata={"category": "policy", "topic": "cancellation"},
        ),
        DocumentInfo(
            id="policy-insurance",
            text="We accept most major insurance providers. Please bring your insurance card to your appointment so we can verify coverage.",
            metadata={"category": "policy", "topic": "insurance"},
        ),
        DocumentInfo(
            id="policy-late",
            text="If you arrive more than 15 minutes late for your appointment, we may need to reschedule you for the next available slot.",
            metadata={"category": "policy", "topic": "late_arrival"},
        ),
        DocumentInfo(
            id="location-1",
            text="We're located at [YOUR ADDRESS HERE]. Parking is available on-site / nearby [update with real details].",
            metadata={"category": "location", "topic": "address"},
        ),
        DocumentInfo(
            id="staff-1",
            text="Our team includes [names/roles here]. You can request a specific staff member when booking, subject to availability.",
            metadata={"category": "staff", "topic": "team"},
        ),
        DocumentInfo(
            id="payment-1",
            text="We accept credit and debit cards, and contactless payments. We do not currently accept checks.",
            metadata={"category": "payment", "topic": "methods"},
        ),
    ]

    try:
        logger.debug("Creating the index...")
        await client.create_index(
            index_name=os.getenv("MOSS_INDEX_NAME"),
            docs=documents,
            model_id="moss-minilm",
        )
        logger.success("Index created successfully.")

    except Exception as e:
        logger.error("An error occurred: {0}", str(e))
        raise


# Run the async function
if __name__ == "__main__":
    asyncio.run(upload_documents())
