"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  BookOpenIcon,
  CpuIcon,
  HistoryIcon,
  LayoutGridIcon,
  ScanIcon,
  UserIcon,
} from "lucide-react"

import { BrandMark, BrandWordmark } from "@/components/brand"
import { SystemStatus } from "@/components/system-status"
import { UserMenu } from "@/components/user-menu"
import { Button } from "@/components/ui/button"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/use-auth"

const MONITOR = [
  { href: "/", label: "Overview", icon: LayoutGridIcon },
  { href: "/diagnoses", label: "History", icon: HistoryIcon },
  { href: "/diseases", label: "Disease library", icon: BookOpenIcon },
]

const MANAGE = [{ href: "/account", label: "Account", icon: UserIcon }]

export function AppSidebar() {
  const pathname = usePathname()
  const { profile } = useAuth()

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href)

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild size="lg" tooltip="Agrivert">
              <Link href="/">
                <BrandMark className="text-primary" />
                <div className="flex flex-col gap-0.5 overflow-hidden">
                  <BrandWordmark />
                  <span className="text-muted-foreground truncate font-mono text-[0.625rem] tracking-wide">
                    vertical farm diagnostics
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            {/* The primary action of the whole product, so it sits above the
                nav rather than inside it. */}
            <Button asChild className="w-full group-data-[collapsible=icon]:hidden">
              <Link href="/diagnose">
                <ScanIcon data-icon="inline-start" />
                Diagnose a plant
              </Link>
            </Button>
            <SidebarMenu className="hidden group-data-[collapsible=icon]:flex">
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  tooltip="Diagnose a plant"
                  isActive={isActive("/diagnose")}
                >
                  <Link href="/diagnose">
                    <ScanIcon />
                    <span>Diagnose</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Monitor</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {MONITOR.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    tooltip={item.label}
                    isActive={isActive(item.href)}
                  >
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Manage</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {MANAGE.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    tooltip={item.label}
                    isActive={isActive(item.href)}
                  >
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
              {/* Gated on the `admin` custom claim — the same claim the API
                  requires, so the nav never offers a route that will 403. */}
              {profile?.isAdmin && (
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    tooltip="Model registry"
                    isActive={isActive("/admin")}
                  >
                    <Link href="/admin">
                      <CpuIcon />
                      <span>Model registry</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SystemStatus />
        <UserMenu />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
