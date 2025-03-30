import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import {StructuredOutputParser } from "@langchain/core/output_parsers";
import {MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import {z} from "zod";
import "dotenv/config";