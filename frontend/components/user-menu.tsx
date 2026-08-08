"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import {
  ChevronsUpDownIcon,
  LogOutIcon,
  MoonIcon,
  ShieldIcon,
  SunIcon,
  UserIcon,
} from "lucide-react"
import { toast } from "sonner"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/use-auth"

export function UserMenu() {
  const { user, profile, signOutUser } = useAuth()
  // Undefined until next-themes resolves on the client, which is exactly the
  // hydration guard we want — no mounted flag needed.
  const { resolvedTheme, setTheme } = useTheme()
  const router = useRouter()
  const isDark = resolvedTheme === "dark"

  const email = profile?.email ?? user?.email ?? ""
  const name = profile?.displayName ?? user?.displayName ?? email.split("@")[0]
  const initials = (name || "?").slice(0, 2).toUpperCase()

  async function handleSignOut() {
    try {
      await signOutUser()
      router.push("/login")
    } catch {
      toast.error("Couldn't sign out. Try again.")
    }
  }

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton size="lg">
              <Avatar className="size-8 rounded-md">
                <AvatarFallback className="rounded-md text-xs">
                  {initials}
                </AvatarFallback>
              </Avatar>
              <div className="flex flex-col gap-0.5 overflow-hidden text-left">
                <span className="truncate text-sm font-medium">{name}</span>
                <span className="text-muted-foreground truncate text-xs">
                  {email}
                </span>
              </div>
              <ChevronsUpDownIcon className="ml-auto" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start" className="w-60">
            <DropdownMenuLabel className="flex flex-col gap-0.5">
              <span className="truncate text-sm">{name}</span>
              <span className="text-muted-foreground truncate text-xs font-normal">
                {email}
              </span>
            </DropdownMenuLabel>
            {profile?.isAdmin && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-muted-foreground flex items-center gap-2 py-1 font-mono text-[0.6875rem] font-normal tracking-wide uppercase">
                  <ShieldIcon className="size-3" />
                  Admin claim
                </DropdownMenuLabel>
              </>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem asChild>
                <Link href="/account">
                  <UserIcon />
                  Account
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={(event) => {
                  event.preventDefault()
                  setTheme(isDark ? "light" : "dark")
                }}
              >
                {isDark ? <SunIcon /> : <MoonIcon />}
                {isDark ? "Light theme" : "Dark theme"}
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={handleSignOut}>
              <LogOutIcon />
              Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
