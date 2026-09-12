import Observer from "../../observer";

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <Observer initialRunId={id} />;
}
