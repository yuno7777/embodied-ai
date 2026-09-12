import type { Metadata } from "next";
import "./styles.css";
export const metadata: Metadata = { title: "Embodied Worlds", description: "Observer for deterministic embodied AI runs" };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
