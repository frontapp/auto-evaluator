import { UseFormReturn } from "react-hook-form";

export type FormValues = {
  evalQuestionsCount: number;
  chunkSize: number;
  overlap: number;
  splitMethod: string;
  embeddingAlgorithm: string;
  model: string;
  retriever: string;
  gradingPrompt: string;
  numNeighbors: number;
  chainMethod: string;
  files: any[];
};

export type Form = UseFormReturn<FormValues>;

export type Result = {
  question: string;
  answer: string;
  result: string;
  retrievalScore: { score: number; justification: string };
  answerScore: { score: number; justification: string };
  latency: number;
  docLinks: string;
};

export type QAPair = {
  question: string;
  answer: string;
};

export type Experiment = {
  evalQuestionsCount: number;
  chunkSize: number;
  overlap: number;
  splitMethod: string;
  retriever: string;
  embeddingAlgorithm: string;
  model: string;
  gradingPrompt: string;
  numNeighbors: number;
  chainMethod: string;
  avgRelevancyScore: number;
  avgAnswerScore: number;
  avgLatency: number;
  performance: number;
  id: number;
};
