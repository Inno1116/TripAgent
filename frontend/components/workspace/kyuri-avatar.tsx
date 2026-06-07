import { cn } from "@/lib/utils";

type KyuriAvatarProps = {
  size?: "sm" | "md" | "lg";
  className?: string;
};

export function KyuriAvatar({ size = "md", className }: KyuriAvatarProps) {
  return (
    <span
      className={cn(
        "relative grid shrink-0 place-items-center overflow-hidden rounded-xl border border-violet-300/35 bg-violet-50 shadow-inner",
        size === "sm" && "h-9 w-9 rounded-lg",
        size === "md" && "h-12 w-12",
        size === "lg" && "h-24 w-24 rounded-2xl",
        className,
      )}
      aria-hidden="true"
    >
      <span className="absolute top-[15%] h-[42%] w-[72%] rounded-[50%_50%_44%_44%] bg-gradient-to-br from-stone-700 via-violet-500 to-rose-300" />
      <span className="absolute bottom-[16%] flex h-[43%] w-[56%] items-center justify-around rounded-[46%_46%_42%_42%] bg-orange-50 shadow-[0_4px_0_rgba(216,111,143,0.12)]">
        <i className="h-[20%] w-[16%] rounded-full bg-stone-950" />
        <i className="h-[20%] w-[16%] rounded-full bg-stone-950" />
      </span>
    </span>
  );
}
