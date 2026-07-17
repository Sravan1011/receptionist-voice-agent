#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Receptionist Voice AI Bot with Moss Semantic Retrieval.

A voice AI receptionist that:
- Answers FAQs (hours, pricing, policies, services) via Moss semantic retrieval
- Supports real-time voice conversations over Pipecat
- Falls back to a human handoff when it doesn't have a confident answer

Required AI services:
- Moss (Semantic Retrieval)
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py

Next steps once this is running (not yet wired in):
- Split FAQ (retrieval) turns from booking turns (calendar tool call) via a
  lightweight intent classifier before the Moss query step.
- Skip the Moss query entirely on low-signal turns ("yes", "okay", "repeat
  that") so retrieval only runs when it's actually needed.
- Tune `top_k` / `alpha` (hybrid keyword+semantic blend) against real call
  transcripts once you have some.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.run import main as runner_main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat_moss import MossRetrievalService

# Load environment variables from .env file
load_dotenv(override=True)

print("Starting Receptionist Voice AI Bot...")
logger.debug("All components loaded successfully!")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the receptionist bot pipeline."""

    # Initialize stt, tts, llm services
    logger.debug("Starting receptionist bot")
    dg_api_key = os.getenv("DEEPGRAM_API_KEY")
    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    assert dg_api_key is not None
    assert cartesia_api_key is not None
    assert openai_api_key is not None

    stt = DeepgramSTTService(api_key=dg_api_key)
    tts = CartesiaTTSService(
        api_key=cartesia_api_key,
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    llm = OpenAILLMService(
        api_key=openai_api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    # Configure Moss retrieval credentials and settings
    project_id = os.getenv("MOSS_PROJECT_ID")
    project_key = os.getenv("MOSS_PROJECT_KEY")
    index_name = os.getenv("MOSS_INDEX_NAME")

    assert project_id is not None
    assert project_key is not None
    assert index_name is not None

    top_k = int(os.getenv("MOSS_TOP_K", "5"))
    # alpha blends keyword and semantic scores: 1.0 = pure semantic,
    # 0.0 = pure keyword. Lower it if callers use a lot of exact terms
    # (service names, staff names) that pure semantic search fuzzes over.
    alpha = float(os.getenv("MOSS_ALPHA", "0.8"))

    moss_service = MossRetrievalService(
        project_id=project_id,
        project_key=project_key,
        system_prompt="Relevant info from the front-desk knowledge base:\n\n",
    )

    # Load the Moss index
    await moss_service.load_index(index_name)
    logger.debug(f"Moss retrieval service initialized (index: {index_name})")

    # System prompt with semantic retrieval support
    agent_name = os.getenv("AGENT_NAME", "Lisa")
    system_content = f"""You are {agent_name}, a friendly, professional \
front-desk receptionist answering phone calls for a business.

Guidelines:
- Your name is {agent_name} — introduce yourself by name when greeting a
  caller, and use it naturally if asked who you are
- Be warm, concise, and conversational — this is a voice call, not a chat window
- Use the provided knowledge base context to answer questions about hours,
  pricing, services, and policies accurately
- If the knowledge base doesn't have a confident answer, say so honestly and
  offer to connect the caller with a staff member — never guess at hours,
  prices, or policy details
- For anything that requires booking, rescheduling, or cancelling an
  appointment, let the caller know you'll get that set up for them
- Ask a clarifying question if the caller's request is unclear

When relevant knowledge base information is provided, use it to give an
accurate, specific answer rather than a generic one."""

    # Initialize conversation context and pipeline components
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
    ]

    context = LLMContext(messages)  # type: ignore
    context_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor()

    # Build the processing pipeline with Moss information injection
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,  # Speech-to-text
            context_aggregator.user(),  # User responses
            moss_service.query(index_name, top_k=top_k, alpha=alpha),  # Moss retrieval
            llm,  # LLM (receives enhanced context)
            tts,  # Text-to-speech
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create and configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Define transport event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.debug("Caller connected")
        # Kick off the conversation with a receptionist greeting
        greeting = (
            "A caller has just connected. Greet them warmly as the front desk "
            "and ask how you can help them today."
        )
        messages.append({"role": "system", "content": greeting})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.debug("Caller disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


# Runner entry point
async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the receptionist bot."""
    # Check required environment variables
    required_vars = [
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "OPENAI_API_KEY",
        "MOSS_PROJECT_ID",
        "MOSS_PROJECT_KEY",
        "MOSS_INDEX_NAME",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("\nPlease update your .env file with the required API keys")
        logger.error("Get your OpenAI API key from: https://platform.openai.com/")
        logger.error("Get your Moss credentials from: https://portal.usemoss.dev")
        return

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    runner_main()
