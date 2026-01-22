#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import websockets   
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, LLMContextFrame, LLMMessagesAppendFrame, EndFrame, EndTaskFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from google.genai.types import HttpOptions


load_dotenv(override=True)
GEMINI_TIMEOUT = 0.5 * 60 * 1000 # 30 seconds

class IdleHandler:
    """Helper class to manage user idle retry logic."""

    def __init__(self):
        self._retry_count = 0

    def reset(self):
        """Reset the retry count when user becomes active."""
        self._retry_count = 0

    async def handle_idle(self, aggregator):
        """Handle user idle event with escalating prompts."""
        self._retry_count += 1

        if self._retry_count == 1:
            # First attempt: Add a gentle prompt to the conversation
            message = {
                "role": "system",
                "content": "The user has been quiet. Politely and briefly ask if they're still there.",
            }
            await aggregator.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
        elif self._retry_count == 2:
            # Second attempt: More direct prompt
            message = {
                "role": "system",
                "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
            }
            await aggregator.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
        elif self._retry_count == 3:
            # Third attempt: End the conversation
            message = {
                "role": "system",
                "content": "Just say, 'It seems like you're busy right now. Have a nice day!'",
            }
            await aggregator.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
        else:
            # Third attempt: Following above message
            await aggregator.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
       

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # set stop_secs to something roughly similar to the internal setting
        # of the Multimodal Live api, just to align events. This doesn't really
        # matter because we can only use the Multimodal Live API's phrase
        # endpointing, for now.
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # set stop_secs to something roughly similar to the internal setting
        # of the Multimodal Live api, just to align events. This doesn't really
        # matter because we can only use the Multimodal Live API's phrase
        # endpointing, for now.
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # set stop_secs to something roughly similar to the internal setting
        # of the Multimodal Live api, just to align events. This doesn't really
        # matter because we can only use the Multimodal Live API's phrase
        # endpointing, for now.
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    session_id = id(transport)  # Unique ID for this bot session
    logger.info(f"🚀 [Session {session_id}] Starting new bot instance")
    
    try:
        logger.info(f"📡 [Session {session_id}] Creating Gemini Live LLM service...")
        llm = GeminiLiveLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Aoede",  # Sulafat
            # system_instruction="Talk like a pirate."
            # inference_on_context_initialization=False,
            http_options=HttpOptions(timeout=GEMINI_TIMEOUT, api_version="v1beta"),
        )
        logger.info(f"✅ [Session {session_id}] Gemini service created successfully")

        logger.info(f"🔧 [Session {session_id}] Setting up LLM context...")
        context = LLMContext(
            [
                {
                    "role": "user",
                    "content": "Say hello. Then introduce yourself as 'Gemini' and ask how you can help the user.",
                },
            ],
        )

        logger.info(f"🔧 [Session {session_id}] Creating aggregators...")
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
                ),
                user_idle_timeout=5.0,  # Detect user idle after 5 seconds
            ),
        )
        logger.info(f"✅ [Session {session_id}] Aggregators created successfully")

        logger.info(f"🔧 [Session {session_id}] Building pipeline...")
        pipeline = Pipeline(
            [
                transport.input(),
                user_aggregator,
                llm,
                transport.output(), # Transport bot output
                assistant_aggregator,
            ]
        )
        logger.info(f"✅ [Session {session_id}] Pipeline built successfully")

        logger.info(f"🔧 [Session {session_id}] Creating pipeline task...")
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )
        logger.info(f"✅ [Session {session_id}] Pipeline task created successfully")

        # Set up idle handling with retry logic
        idle_handler = IdleHandler()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"👤 [Session {session_id}] Client connected - starting conversation")
            # Kick off the conversation.
            await task.queue_frames([LLMRunFrame()])
            logger.info(f"✅ [Session {session_id}] Initial LLMRunFrame queued")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"👋 [Session {session_id}] Client disconnected - starting cleanup")
            try:
                # Cancel the task when client disconnects
                await task.cancel()
                logger.info(f"✅ [Session {session_id}] Task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ [Session {session_id}] Error cancelling task: {e}", exc_info=True)

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
            timestamp = f"[{message.timestamp}] " if message.timestamp else ""
            line = f"{timestamp}user: {message.content}"
            logger.info(f"Transcript: {line}")

        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            timestamp = f"[{message.timestamp}] " if message.timestamp else ""
            line = f"{timestamp}assistant: {message.content}"
            logger.info(f"Transcript: {line}")

        @user_aggregator.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(aggregator):
            await idle_handler.handle_idle(aggregator)

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            idle_handler.reset()

        logger.info(f"🏃 [Session {session_id}] Creating pipeline runner...")
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        logger.info(f"✅ [Session {session_id}] Pipeline runner created, starting execution...")

        await runner.run(task)
        
        logger.info(f"🏁 [Session {session_id}] Pipeline run completed normally")
        
    except Exception as e:
        logger.error(f"❌ [Session {session_id}] Error in run_bot: {e}", exc_info=True)
        raise
    finally:
        logger.info(f"🧹 [Session {session_id}] Bot instance cleanup complete")


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    # Create a new transport for each connection
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()