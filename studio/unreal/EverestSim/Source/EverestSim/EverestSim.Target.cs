using UnrealBuildTool;
using System.Collections.Generic;

public class EverestSimTarget : TargetRules
{
    public EverestSimTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_6;
        ExtraModuleNames.Add("EverestSim");
    }
}
