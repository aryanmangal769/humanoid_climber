using UnrealBuildTool;

public class EverestSim : ModuleRules
{
    public EverestSim(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] {
            "Core", "CoreUObject", "Engine", "InputCore", "Json", "JsonUtilities",
            "ProceduralMeshComponent", "WebSockets", "Sockets", "Networking"
        });
    }
}
