"use client";

import { useEffect, useState } from "react";

type EvaluationSummary = {
  total_questions: number;
  evaluation_type?: string;
  precision_at_k?: Record<string, number>;
  recall_at_k?: Record<string, number>;
  f1_at_k?: number;
  hit_at_1?: number;
  hit_at_3?: number;
  hit_at_5?: number;
  mrr?: number;
  source_accuracy?: number;
  section_accuracy?: number;
  refusal_accuracy?: number;
  avg_retrieval_latency_ms: number;
};

type QuestionResult = {
  id: string;
  question: string;
  query_type?: string;
  expected_answer?: string | null;
  answer?: string;
  precision_at_k?: Record<string, number>;
  recall_at_k?: Record<string, number>;
  first_relevant_rank?: number | null;
  refused?: boolean | null;
  retrieval_latency_ms?: number;
  llm_latency_ms?: number;
  error?: string;
};

type EvaluationData = {
  summary: EvaluationSummary;
  questions: QuestionResult[];
  history?: { run_id: string; created_at: string; summary: EvaluationSummary }[];
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function MetricCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5">
      <p className="text-sm text-zinc-400">
        {title}
      </p>

      <p className="mt-2 text-3xl font-semibold text-white">
        {value}
      </p>
    </div>
  );
}

export default function EvaluationPage() {
  const [data, setData] =
    useState<EvaluationData | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  async function loadEvaluation() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `${API_URL}/api/evaluation`
      );

      if (!response.ok) {
        throw new Error(
          "Failed to load evaluation"
        );
      }

      const result =
        await response.json();

      if (result.message) {
        setData(null);
        setError(result.message);
        return;
      }

      setData(result);

    } catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong"
      );

    } finally {

      setLoading(false);

    }
  }

  useEffect(() => {
    loadEvaluation();
  }, []);

  if (loading) {
    return (
      <main className="min-h-screen bg-black p-8 text-white">
        <p>Loading evaluation...</p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-black p-8 text-white">
        <h1 className="text-3xl font-bold">
          RAG Evaluation
        </h1>

        <p className="mt-4 text-zinc-400">
          {error || "No evaluation data available."}
        </p>
      </main>
    );
  }

  const { summary, questions } = data;
  const precisionAtK = summary.precision_at_k?.["6"] ?? 0;
  const recallAtK = summary.recall_at_k?.["6"] ?? 0;

  return (
    <main className="min-h-screen bg-black px-6 py-10 text-white md:px-12">

      <div className="mx-auto max-w-7xl">

        <div className="mb-10 flex flex-col justify-between gap-4 md:flex-row md:items-end">

          <div>
            <p className="text-sm uppercase tracking-widest text-zinc-500">
              Resume RAG
            </p>

            <h1 className="mt-2 text-4xl font-bold">
              Evaluation Dashboard
            </h1>

            <p className="mt-2 text-zinc-400">
              Deterministic retrieval-only evaluation. Groq is not called,
              so provider rate limits cannot invalidate these measurements.
            </p>
          </div>

          <button
            onClick={loadEvaluation}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm hover:bg-zinc-900"
          >
            Refresh
          </button>

        </div>

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

          <MetricCard
            title="Precision@6"
            value={percentage(precisionAtK)}
          />

          <MetricCard
            title="Recall@6"
            value={percentage(recallAtK)}
          />

          <MetricCard
            title="F1@6"
            value={percentage(summary.f1_at_k ?? 0)}
          />

          <MetricCard
            title="Hit@3"
            value={percentage(summary.hit_at_3 ?? 0)}
          />

        </section>

        <section className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

          <MetricCard
            title="Questions"
            value={String(
              summary.total_questions
            )}
          />

          <MetricCard
            title="MRR"
            value={percentage(summary.mrr ?? 0)}
          />

          <MetricCard
            title="Avg Retrieval"
            value={`${summary.avg_retrieval_latency_ms} ms`}
          />

          <MetricCard
            title="Refusal Accuracy"
            value={percentage(summary.refusal_accuracy ?? 0)}
          />

        </section>

        {data.history && data.history.length > 0 && (
          <section className="mt-8 rounded-xl border border-zinc-800 bg-zinc-950 p-5">
            <h2 className="text-lg font-semibold">Run History</h2>
            <div className="mt-3 flex flex-wrap gap-3 text-sm text-zinc-400">
              {data.history.slice(-5).reverse().map((run) => (
                <span key={run.run_id} className="rounded-lg border border-zinc-800 px-3 py-2">
                  {new Date(run.created_at).toLocaleString()} · P@6 {percentage(run.summary.precision_at_k?.["6"] ?? 0)}
                </span>
              ))}
            </div>
          </section>
        )}

        <section className="mt-10">

          <div className="mb-4">
            <h2 className="text-2xl font-semibold">
              Question Results
            </h2>

            <p className="mt-1 text-sm text-zinc-500">
              Detailed performance for every
              evaluation question.
            </p>
          </div>

          <div className="overflow-hidden rounded-xl border border-zinc-800">

            <div className="overflow-x-auto">

              <table className="w-full text-left text-sm">

                <thead className="border-b border-zinc-800 bg-zinc-900">

                  <tr>
                    <th className="px-5 py-4">
                      Question
                    </th>

                    <th className="px-5 py-4">
                      Type
                    </th>

                    <th className="px-5 py-4">
                      Precision
                    </th>

                    <th className="px-5 py-4">
                      Recall
                    </th>

                    <th className="px-5 py-4">
                      First Hit
                    </th>

                    <th className="px-5 py-4">
                      Refused
                    </th>
                  </tr>

                </thead>

                <tbody>

                  {questions.map((item) => (

                    <tr
                      key={item.id}
                      className="border-b border-zinc-900 hover:bg-zinc-950"
                    >

                      <td className="max-w-md px-5 py-4 text-zinc-200">
                        {item.question}
                      </td>

                      <td className="px-5 py-4 text-zinc-400">
                        {item.query_type || "-"}
                      </td>

                      <td className="px-5 py-4">
                        {item.precision_at_k?.["6"] !== undefined
                          ? percentage(
                              item.precision_at_k["6"]
                            )
                          : "-"}
                      </td>

                      <td className="px-5 py-4">
                        {item.recall_at_k?.["6"] !== undefined
                          ? percentage(
                              item.recall_at_k["6"]
                            )
                          : "-"}
                      </td>

                      <td className="px-5 py-4">
                        {item.first_relevant_rank ?? "-"}
                      </td>

                      <td className="px-5 py-4">
                        {item.refused === null || item.refused === undefined
                          ? "-"
                          : item.refused ? "Yes" : "No"}
                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          </div>

        </section>

      </div>

    </main>
  );
}
